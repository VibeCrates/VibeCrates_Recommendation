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

from typing import Iterable, Sequence

# HF 경로(generate_item_descriptions.generate_qwen)와 동일하게 맞춘 디코딩 설정.
# 둘 사이에서 출력이 달라지면 품질 비교가 불가능해지므로 여기서만 바꾼다.
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_REPETITION_PENALTY = 1.2


class VLLMRunner:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
    ):
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

    def generate(self, items: Sequence[tuple[str, object | None]]) -> list[str]:
        """items = [(prompt, PIL.Image or None), ...] → 생성 문자열 목록 (같은 순서)."""
        if not items:
            return []
        requests: list[dict] = []
        for prompt, image in items:
            req: dict = {"prompt": self._chat_text(prompt, image is not None)}
            if image is not None:
                req["multi_modal_data"] = {"image": image}
            requests.append(req)
        outputs = self.llm.generate(requests, self.params)
        return [o.outputs[0].text.strip() for o in outputs]


def chunks(seq: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
