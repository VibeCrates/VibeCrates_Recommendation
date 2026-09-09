"""평가용 한국어 쿼리 40개를 번역해 JSON으로 남긴다 (opus-mt와 Qwen 두 벌).

왜 필요한가:
  쿼리 인코더가 CLIP 텍스트 인코더(영어 BPE)라 한글을 바이트 조각으로 부순다. 한국어를
  그대로 넣으면 쿼리 벡터가 영어 등가 문장과 거의 직교하고(코사인 0.058), 관련성 판정도
  0.130까지 떨어진다(영어 0.766). 지금 서빙은 opus-mt-ko-en으로 번역해 넘기는데 은유에서
  흔들린다 — 스타일별 코사인이 poet만 0.632다("하늘"이 "Heaven"으로 번역된다).

  이 스크립트는 그 번역기를 Qwen으로 바꿨을 때 얼마나 회복되는지 재기 위한 재료를 만든다.
  번역과 검색을 한 프로세스에 두지 않는 것은 vLLM이 GPU를 90% 잡아 추천 모델을 함께
  올릴 수 없기 때문이다.

사용:
  python scripts/translate_eval_queries.py            # 둘 다
  python scripts/translate_eval_queries.py --only opus
"""
import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def load_queries() -> list:
    """eval_lang.py의 QUERIES를 **임포트하지 않고** 소스에서 읽는다.

    이 스크립트는 vLLM용 venv에서 돌고 거기에는 sentence_transformers가 없다(학습 환경과
    분리돼 있다). eval_lang을 임포트하면 추천 모델까지 끌려온다. 목록을 복사해 두면
    원본과 어긋나므로 파싱한다.
    """
    tree = ast.parse((_ROOT / "scripts" / "eval_lang.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "QUERIES" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("eval_lang.py에서 QUERIES를 찾지 못했다")

OUT = "data/cache/eval_query_translations.json"

# 은유를 살리라고 명시한다. opus-mt의 실패가 "하늘 → Heaven"처럼 어휘를 종교적·문자적으로
# 좁히는 형태였으므로, 그 반대 방향을 프롬프트에 박아 둔다.
PROMPT = (
    "[Task]\n"
    "Translate this Korean search query into natural English. It is a search query for a "
    "recommendation service, often evocative or metaphorical.\n\n"
    "[Korean]\n{ko}\n\n"
    "[Rules]\n"
    "- Keep the imagery and mood. Do not resolve a metaphor into a literal or religious term.\n"
    "- Keep it as a short phrase, not a sentence with a subject and verb.\n"
    "- Do not add words that are not implied by the Korean.\n\n"
    "[Output]\nThe English phrase only. No quotes, no explanation."
)


def clean(raw: str) -> str:
    s = str(raw).strip().split("\n")[0].strip()
    s = re.sub(r'^["\'“‘]|["\'”’]$', "", s).strip()
    return re.sub(r"^(English|Translation)\s*:\s*", "", s, flags=re.I).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["opus", "qwen"])
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ko = [(f"{c}{p}_{l.upper()}", q) for c, p, l, q in load_queries() if l == "ko"]
    print(f"한국어 쿼리 {len(ko)}개")

    out = json.load(open(args.out)) if os.path.exists(args.out) else {}

    if args.only != "qwen":
        from src.api.translation import QueryTranslator
        tr = QueryTranslator()
        out["opus"] = dict(zip([k for k, _ in ko], tr.translate([q for _, q in ko])))
        print("opus-mt 완료")

    if args.only != "opus":
        from scripts.vllm_runner import VLLMRunner
        runner = VLLMRunner(args.model_id, max_new_tokens=64, repetition_penalty=1.0)
        raw = runner.generate([(PROMPT.format(ko=q), None) for _, q in ko])
        out["qwen"] = {k: clean(r) for (k, _), r in zip(ko, raw)}
        print("Qwen 완료")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    for k, q in ko:
        print(f"  {k:8s} {q:24s} | opus: {out.get('opus', {}).get(k, '-'):38s} | qwen: {out.get('qwen', {}).get(k, '-')}")
    print(f"\n저장 → {args.out}")


if __name__ == "__main__":
    main()
