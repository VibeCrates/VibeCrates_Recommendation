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
"""
import logging
import os
import re
import threading

logger = logging.getLogger(__name__)

# 한글 음절. 자모(ㄱ-ㅎ, ㅏ-ㅣ)만 있는 입력은 번역해도 의미가 없으므로 제외한다.
HANGUL = re.compile(r"[가-힣]")

MT_MODEL   = os.getenv("MT_MODEL", "Helsinki-NLP/opus-mt-ko-en")
MT_ENABLED = os.getenv("TRANSLATE_QUERIES", "1").lower() not in ("0", "", "false", "no")
MT_BEAMS   = int(os.getenv("MT_BEAMS", "4"))


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

        targets = [i for i, t in enumerate(texts) if self.needs_translation(t)]
        if not targets or not self._ensure_loaded():
            return list(texts)

        try:
            batch = self._tokenizer([texts[i] for i in targets],
                                    return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                out = self._model.generate(**batch, max_new_tokens=64, num_beams=MT_BEAMS)
            decoded = self._tokenizer.batch_decode(out, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"번역 실패 — 원문을 그대로 인코딩합니다: {e}")
            return list(texts)

        result = list(texts)
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
