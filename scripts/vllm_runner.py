"""
Qwen2.5-VL 배치 추론 러너 (vLLM).

왜 필요한가:
  기존 HF `model.generate` 경로는 batch=1 greedy다. RTX 5090 실측 1.48 s/it
  (movie 200건 4분 55초, GPU util 91%지만 VRAM은 16.8/32GB만 사용).
  description_synth 190,739건 + 쿼리 92,447건 = 약 28만 회를 그 속도로 돌리면
  116시간(4.8일)이다. vLLM의 continuous batching으로 프롬프트와 디코딩 설정은
  그대로 둔 채 처리량만 올린다.

호출 계약은 HF 경로와 동일하게 맞춘다 — (프롬프트, PIL 이미지 or None) 목록을 받아
생성 문자열 목록을 **입력과 같은 순서로** 돌려준다. 이미지가 없는 항목은 텍스트 전용
요청으로 나가므로, 없는 포스터를 근거로 삼으라는 환각 유도가 생기지 않는다
(프롬프트 쪽 grounding 분기는 호출자가 이미 처리한다).

주의: vLLM은 자체 torch 빌드를 가져오므로 학습용 venv와 분리한 venv에 설치한다.
  /mnt/data8tb/friend/vibecrates/venv_vllm/bin/python
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

# HF 경로(generate_item_descriptions.generate_qwen)와 동일하게 맞춘 디코딩 설정.
# 둘 사이에서 출력이 달라지면 품질 비교가 불가능해지므로 여기서만 바꾼다.
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_REPETITION_PENALTY = 1.2

# 이미지 토큰 상한 (2026-08-05 stage2 전멸의 원인).
#   Qwen2.5-VL 프로세서 기본값은 size.longest_edge=12,845,056px = 시각 토큰 4,096개다.
#   즉 큰 표지 한 장이 컨텍스트(4,096) 전체를 텍스트 한 글자 없이 먹는다. 실제로
#   "decoder prompt (length 4268) is longer than max_model_len 4096"으로 배치 전체가
#   죽었다. 프로세서 설정(transformers 5.x에서 min/max_pixels → size dict로 이동)에
#   기대지 않고, 넘기기 전에 우리가 직접 줄여 상한을 보장한다.
#   1,280 토큰은 Qwen 문서 권장 상한이며 포스터·표지의 mood 판독에는 충분하다.
MAX_IMAGE_TOKENS = 1280
IMAGE_FACTOR = 28          # patch_size(14) * merge_size(2)
IMAGE_PIXEL_CAP = MAX_IMAGE_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR
# chat template의 특수 토큰·role 헤더 등 프롬프트 본문 밖 여유분.
TEMPLATE_RESERVE = 96


class VLLMRunner:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
    ):
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        # 프롬프트를 chat template로 펴는 데만 쓴다 (토크나이즈는 vLLM이 한다).
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1},
        )
        # temperature=0 = greedy. HF 경로의 do_sample=False와 같은 결정론적 디코딩.
        self.params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )

    def _chat_text(self, prompt: str, has_image: bool) -> str:
        content: list[dict] = []
        if has_image:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    # ── 길이 방어 ─────────────────────────────────────────────────────────────
    # vLLM은 요청 하나가 max_model_len을 넘으면 add_request에서 ValueError를 던지고,
    # 그 예외가 배치 전체(512건)와 프로세스를 같이 죽인다. 애초에 넘기지 않는 게 1차
    # 방어, 그래도 터지면 배치를 쪼개 문제 항목만 버리는 게 2차 방어다.

    @staticmethod
    def _fit_image(image):
        """시각 토큰이 MAX_IMAGE_TOKENS를 넘지 않도록 원본을 미리 축소한다."""
        w, h = image.size
        if w * h <= IMAGE_PIXEL_CAP:
            return image
        scale = math.sqrt(IMAGE_PIXEL_CAP / (w * h))
        from PIL import Image as _Image
        return image.resize(
            (max(IMAGE_FACTOR, int(w * scale)), max(IMAGE_FACTOR, int(h * scale))),
            _Image.LANCZOS,
        )

    @staticmethod
    def _image_tokens(image) -> int:
        """축소 후 이미지가 차지할 토큰 수 (Qwen smart_resize의 반올림까지 반영).
        서버 실측: 1000x1500 → 1,966 / 1650x1650 → 3,503 토큰 (≈ 픽셀/784)."""
        w, h = image.size
        hb = max(IMAGE_FACTOR, round(h / IMAGE_FACTOR) * IMAGE_FACTOR)
        wb = max(IMAGE_FACTOR, round(w / IMAGE_FACTOR) * IMAGE_FACTOR)
        return (hb // IMAGE_FACTOR) * (wb // IMAGE_FACTOR)

    def _fit_prompt(self, prompt: str, image_tokens: int) -> str:
        """남은 예산을 넘는 프롬프트는 가운데를 들어낸다.
        머리(Task·Rules)와 꼬리(Output 지시)는 지시문이라 자르면 출력 형식이 깨진다."""
        budget = self.max_model_len - self.max_new_tokens - image_tokens - TEMPLATE_RESERVE
        tok = self.processor.tokenizer
        ids = tok.encode(prompt, add_special_tokens=False)
        if len(ids) <= budget or budget <= 0:
            return prompt
        head, tail = int(budget * 0.6), budget - int(budget * 0.6)
        return (
            tok.decode(ids[:head]) + "\n…(중략)…\n" + tok.decode(ids[-tail:])
        )

    def _generate_requests(self, requests: list[dict]) -> list[str]:
        """실패하면 이분 분할로 좁혀 문제 항목만 빈 문자열로 떨어뜨린다.
        (캐시에 안 들어가므로 다음 실행에서 자동 재시도된다.)"""
        try:
            outputs = self.llm.generate(requests, self.params)
            return [o.outputs[0].text.strip() for o in outputs]
        except Exception as e:
            if len(requests) == 1:
                print(f"  [warn] 요청 1건 건너뜀: {e}", flush=True)
                return [""]
            mid = len(requests) // 2
            print(f"  [warn] 배치 {len(requests)}건 실패 → 분할 재시도: {e}", flush=True)
            return self._generate_requests(requests[:mid]) + self._generate_requests(requests[mid:])

    def generate(self, items: Sequence[tuple[str, object | None]]) -> list[str]:
        """items = [(prompt, PIL.Image or None), ...] → 생성 문자열 목록 (같은 순서)."""
        if not items:
            return []
        requests: list[dict] = []
        for prompt, image in items:
            image = self._fit_image(image) if image is not None else None
            img_tokens = self._image_tokens(image) if image is not None else 0
            req: dict = {
                "prompt": self._chat_text(self._fit_prompt(prompt, img_tokens), image is not None)
            }
            if image is not None:
                req["multi_modal_data"] = {"image": image}
            requests.append(req)
        return self._generate_requests(requests)


def chunks(seq: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
