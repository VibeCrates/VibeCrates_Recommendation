"""한국어 검색어를 영어로 번역해 인코더에 넘긴다.

왜 필요한가 (2026-08-31 실측)
  쿼리 인코더는 CLIP 텍스트 인코더인데 영어 BPE라 한글을 바이트 조각으로 부순다.
  "비 오는 오후의 조용한 시간"이 ë¹/Ħ/ìĺ/¤ 같은 27개 조각이 되고, 학습 쿼리도 콘텐츠도
  전부 영어라 모델이 한글을 본 적이 없다. 결과는 이렇다.

    같은 뜻의 한/영 쿼리 40쌍 · 쿼리 벡터 코사인
      한국어 그대로  0.058   ← 영어 등가 문장과 거의 직교. 의미가 담기지 않는다
      번역 후        0.780
    검색 최고 점수
      영어 0.659 / 한국어 그대로 0.509 / 번역 후 0.645

  한국어 결과가 0.455~0.509 좁은 구간에 몰리는 것은 쿼리 벡터가 의미를 담지 못하고
  공간의 평균 근처에 떨어질 때 나오는 모양이다. 상위 10개 겹침도 40쌍 전부 0%였다.

근본 해결은 다국어 인코더로 바꾸는 것이고(백로그 A2와 같은 줄기) 재학습이 필요하다.
이 모듈은 그때까지의 다리다 — 재학습 없이, 백엔드 변경 없이 넣을 수 있다.

한계: 은유적 표현에서 번역이 흔들린다. 스타일별 코사인이 atmosphere 0.848 /
direct 0.857 / philosophical 0.783인데 poet만 0.632다.

번역에 실패해도 요청을 죽이지 않는다. 원문을 그대로 인코딩하고 넘어간다 — 품질이
떨어질 뿐 검색은 되며, 번역기 하나 때문에 서비스가 멈추는 편이 더 나쁘다.

2026-10-12 — 사전을 먼저 본다:
  백엔드가 실제로 보내는 검색어는 "시집", "로파이 비 오는 날" 같은 짧은 단어 조합이고,
  opus-mt는 거기서 무너진다. 로그 530종 중 33.0%가 핵심어를 잃었다(시집 → "Indulgent",
  보사노바 → "Bosanoba"). 그래서 검색어를 사전(ko_en_dict.json, scripts/build_translation_dict.py)
  의 단어로 끝까지 쪼갤 수 있으면 사전 번역을 이어 붙이고, 하나라도 못 쪼개면 opus-mt로
  넘긴다. 사전은 어휘 단위로만 맞으므로 문장형 검색어는 지금처럼 opus-mt가 옮긴다.

2026-10-23 — 사전으로 안 풀리는 문장형은 LLM에 먼저 보낸다:
  10/15 사람 판정에서 opus-mt는 평균 3.64점·1~2점 17.1%, Gemma 3 4B는 4.42~4.52점·
  0.7~2.9%였다(experiments/translation_review_20261015_*). GPU가 있는 서버(학교 AstraGo)에서
  vLLM으로 띄운 Gemma를 HTTP로 부른다 — 문장당 0.13초, 20건 동시 0.33초. LLM_TRANSLATE_URL이
  없으면(Mac 서버) 이 단계를 건너뛰고, 시간 안에 답이 없거나 실패한 검색어만 opus-mt로 넘긴다.
"""
import json
import logging
import os
import re
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

# 한글 음절. 자모(ㄱ-ㅎ, ㅏ-ㅣ)만 있는 입력은 번역해도 의미가 없으므로 제외한다.
HANGUL = re.compile(r"[가-힣]")

MT_MODEL   = os.getenv("MT_MODEL", "Helsinki-NLP/opus-mt-ko-en")
MT_ENABLED = os.getenv("TRANSLATE_QUERIES", "1").lower() not in ("0", "", "false", "no")
MT_BEAMS   = int(os.getenv("MT_BEAMS", "4"))
DICT_PATH  = os.getenv("KO_EN_DICT", str(Path(__file__).with_name("ko_en_dict.json")))

# OpenAI 호환 서버 주소(예: http://127.0.0.1:8001/v1). 비어 있으면 LLM 번역을 쓰지 않는다.
LLM_URL     = os.getenv("LLM_TRANSLATE_URL", "").rstrip("/")
LLM_MODEL   = os.getenv("LLM_TRANSLATE_MODEL", "gemma3-4b")
LLM_TIMEOUT = float(os.getenv("LLM_TRANSLATE_TIMEOUT", "1.0"))

# scripts/translate_sentence_bench.py의 PROMPT와 같은 문구 — 10/15 판정이 이 프롬프트로 나왔다.
LLM_PROMPT = (
    "Translate this Korean search query from a book/movie/music recommendation app into "
    "natural English.\n\n"
    "Rules:\n"
    "- Keep the meaning, imagery and mood. Do not turn a metaphor into a literal or religious term.\n"
    "- Do not add anything the Korean does not say, and do not drop anything it does say.\n"
    "- Slang and internet abbreviations (ㅊㅊ = recommend, ㄱㄱ = go) become plain English.\n"
    "- Titles and names use their usual English form.\n\n"
    "Korean: {ko}\n\n"
    "Output the English translation only. No quotes, no explanation."
)

# 사전 단어 뒤에 붙은 조사는 떼고 찾는다("바다의" → 바다). 긴 것부터 시도한다.
# "처럼"·"같은"은 넣지 않는다 — "영화 같은 노래"를 "movie songs"로 붙이면 뜻이 바뀐다.
PARTICLES = ("에서", "으로", "이랑", "의", "에", "을", "를", "은", "는", "이", "가",
             "와", "과", "도", "로", "랑")


class DictTranslator:
    """검색어를 사전 단위로 끝까지 쪼갤 수 있을 때만 번역한다. 못 쪼개면 None."""

    def __init__(self, path: str = DICT_PATH):
        self.entries: dict[str, str] = {}
        try:
            with open(path, encoding="utf-8") as f:
                self.entries = json.load(f)["entries"]
            logger.info(f"번역 사전 {len(self.entries):,}개 적재: {path}")
        except Exception as e:
            # 사전이 없으면 opus-mt만으로 지금처럼 돈다.
            logger.error(f"번역 사전 적재 실패 — opus-mt만 씁니다: {e}")
        self.max_words = max((len(k.split()) for k in self.entries), default=0)

    def _word(self, token: str) -> str | None:
        key = token.lower()
        if key in self.entries:
            return self.entries[key]
        # 조사 떼기는 남는 말이 두 글자 이상일 때만 — "시가"를 시(poetry)+가로 읽지 않게
        for p in PARTICLES:
            stem = key[:-len(p)]
            if key.endswith(p) and len(stem) >= 2 and stem in self.entries:
                return self.entries[stem]
        if not HANGUL.search(token):
            return token          # 숫자·영어 단어는 그대로 둔다("재즈 3", "BTS 발라드")
        return None

    def translate(self, text: str) -> str | None:
        toks = text.split()
        out, i = [], 0
        while i < len(toks):
            # 여러 어절짜리 단위("비 오는 날", "주말 오후")를 먼저 본다
            for n in range(min(self.max_words, len(toks) - i), 1, -1):
                phrase = " ".join(toks[i:i + n]).lower()
                if phrase in self.entries:
                    out.append(self.entries[phrase])
                    i += n
                    break
            else:
                w = self._word(toks[i])
                if w is None:
                    return None
                out.append(w)
                i += 1
        result = " ".join(w for w in out if w)
        return result or None


class LLMTranslator:
    """vLLM 서버에 한 검색어씩 동시에 보낸다. vLLM이 동시 요청을 묶어 처리한다."""

    def __init__(self, url: str = LLM_URL):
        self.url = url
        self._pool = ThreadPoolExecutor(max_workers=16) if url else None
        if url:
            logger.info(f"LLM 번역 사용: {url} ({LLM_MODEL}, 제한 {LLM_TIMEOUT}s)")

    @staticmethod
    def _clean(raw: str) -> str:
        s = raw.strip().split("\n")[0].strip()
        s = re.sub(r"^(English|Translation)\s*:\s*", "", s, flags=re.I).strip()
        return re.sub(r'^["\'“‘]|["\'”’]$', "", s).strip()

    def _one(self, text: str) -> str | None:
        body = json.dumps({
            "model": LLM_MODEL, "temperature": 0, "max_tokens": 96,
            "messages": [{"role": "user", "content": LLM_PROMPT.format(ko=text)}],
        }).encode()
        req = urllib.request.Request(f"{self.url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
                out = self._clean(json.load(r)["choices"][0]["message"]["content"])
        except Exception as e:
            logger.warning(f"LLM 번역 실패({type(e).__name__}) — opus-mt로 넘깁니다: {text!r}")
            return None
        # 번역문에 한글이 남으면 옮기지 못한 것이다
        return out if out and not HANGUL.search(out) else None

    def translate(self, texts: list[str]) -> list[str | None]:
        if not self._pool:
            return [None] * len(texts)
        return list(self._pool.map(self._one, texts))


class QueryTranslator:
    """모델은 첫 한국어 요청 때 올린다.

    기동 시 올리면 한국어를 한 번도 안 쓰는 환경에서도 300MB를 받고 몇 초를 쓴다.
    /ping을 모델과 분리해 둔 것과 같은 취지다.
    """

    def __init__(self, model_name: str = MT_MODEL):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._failed = False          # 한 번 실패하면 매 요청마다 재시도하지 않는다
        self.dictionary = DictTranslator()
        self.llm = LLMTranslator()

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                from transformers import MarianMTModel, MarianTokenizer
                logger.info(f"번역 모델 적재: {self.model_name}")
                self._tokenizer = MarianTokenizer.from_pretrained(self.model_name)
                self._model = MarianMTModel.from_pretrained(self.model_name)
                self._model.eval()
                logger.info("번역 모델 준비 완료.")
                return True
            except Exception as e:
                self._failed = True
                logger.error(
                    f"번역 모델 적재 실패 — 한국어 검색어가 번역 없이 인코딩됩니다: {e}"
                )
                return False

    @staticmethod
    def needs_translation(text: str) -> bool:
        return bool(HANGUL.search(text or ""))

    def translate(self, texts: list[str]) -> list[str]:
        """한글이 든 것만 번역해 돌려준다. 나머지는 원문 그대로."""
        import torch

        result = list(texts)
        targets = []
        for i, t in enumerate(texts):
            if not self.needs_translation(t):
                continue
            hit = self.dictionary.translate(t)
            if hit is not None:
                logger.info(f"번역(사전): {t!r} → {hit!r}")
                result[i] = hit
            else:
                targets.append(i)
        if targets and self.llm.url:
            remaining = []
            for i, hit in zip(targets, self.llm.translate([texts[i] for i in targets])):
                if hit is None:
                    remaining.append(i)
                else:
                    logger.info(f"번역(LLM): {texts[i]!r} → {hit!r}")
                    result[i] = hit
            targets = remaining
        # 사전·LLM으로 다 풀리면 opus-mt를 올리지도 않는다
        if not targets or not self._ensure_loaded():
            return result

        try:
            batch = self._tokenizer([texts[i] for i in targets],
                                    return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                out = self._model.generate(**batch, max_new_tokens=64, num_beams=MT_BEAMS)
            decoded = self._tokenizer.batch_decode(out, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"번역 실패 — 원문을 그대로 인코딩합니다: {e}")
            return result

        for i, translated in zip(targets, decoded):
            translated = translated.strip()
            if translated:
                logger.info(f"번역: {texts[i]!r} → {translated!r}")
                result[i] = translated
        return result


_translator = QueryTranslator()


def prepare(texts: list[str]) -> tuple[list[str], bool]:
    """인코더에 넣을 문자열과, 하나라도 번역됐는지 여부.

    번역문을 응답에 실어 백엔드가 '무엇으로 검색됐는지' 볼 수 있게 하려고 bool을 함께
    돌려준다. 결과가 이상할 때 번역 탓인지 검색 탓인지 가르는 유일한 단서다.
    """
    if not MT_ENABLED:
        return list(texts), False
    used = _translator.translate(texts)
    return used, used != list(texts)
