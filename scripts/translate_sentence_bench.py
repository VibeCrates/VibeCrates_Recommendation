"""문장형 한국어 검색어를 LLM 번역기 후보로 옮겨 JSON으로 남긴다 (beast, vLLM).

왜 필요한가 (2026-10-12):
  번역 사전(ko_en_dict.json)은 "시집", "로파이 비 오는 날" 같은 단어 조합만 푼다. 평가 문장
  40개 중 사전으로 풀린 것은 1개뿐이고 나머지는 opus-mt로 가는데, opus-mt는 은유 문장에서
  무너진다(poet 코사인 0.632). 문장형 검색어를 받으려면 LLM급 번역기가 필요하고, 남은 문제는
  어느 크기의 모델을 어디서 돌리느냐다. 이 스크립트는 후보별 번역만 만들고, 채점은 서빙
  모델이 있는 Mac에서 scripts/eval_sentence_translation.py가 한다.

입력:
  data/translation/sentence_bench.tsv  — 실사용형 문장 100개와 기준 영어
  scripts/eval_lang.py QUERIES         — 평가 쿼리 40쌍(한국어와 사람이 쓴 영어 원문)

모델 하나에 프로세스 하나다. vLLM은 한 프로세스에서 모델을 내리고 다시 올릴 때 GPU 메모리를
깨끗이 돌려주지 않는다.

사용 (beast, source env.sh 후):
  ../venv_vllm/bin/python scripts/translate_sentence_bench.py --tag qwen25_3b
"""
import argparse
import ast
import csv
import json
import os
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
BENCH = _ROOT / "data/translation/sentence_bench.tsv"
OUT_DIR = _ROOT / "experiments/sentence_translation"

# tag → (모델 id, 필요한 GPU 메모리 GB). beast GPU는 다른 사용자와 나눠 쓰므로 필요한 만큼만 잡는다.
MODELS = {
    "gemma3_27b":  ("RedHatAI/gemma-3-27b-it-quantized.w4a16", 20),
    "qwen25vl_7b": ("Qwen/Qwen2.5-VL-7B-Instruct", 19),
    "qwen3_4b":    ("Qwen/Qwen3-4B-Instruct-2507", 10),
    "gemma3_4b":   ("unsloth/gemma-3-4b-it", 10),       # google/ 원본은 동의 절차가 있어 토큰 없이 못 받는다
    "qwen25_3b":   ("Qwen/Qwen2.5-3B-Instruct", 8),
    "kanana_2b":   ("kakaocorp/kanana-1.5-2.1b-instruct-2505", 7),
    "qwen25_1b":   ("Qwen/Qwen2.5-1.5B-Instruct", 5),
}

# 모든 후보에 같은 프롬프트. 문장은 문장으로 옮기게 둔다 — 기준 영어도 문장이다.
PROMPT = (
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


def load_eval_queries() -> list[tuple[str, str, str]]:
    """eval_lang.py의 QUERIES를 임포트하지 않고 소스에서 읽는다 (vLLM venv에 추천 모델 의존성이 없다)."""
    tree = ast.parse((_ROOT / "scripts" / "eval_lang.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "QUERIES" for t in node.targets):
            q = ast.literal_eval(node.value)
            en = {(c, p): t for c, p, l, t in q if l == "en"}
            return [(f"eval_{c}", k, en[(c, p)]) for c, p, l, k in q if l == "ko"]
    raise RuntimeError("eval_lang.py에서 QUERIES를 찾지 못했다")


def load_items() -> list[dict]:
    with open(BENCH, encoding="utf-8") as f:
        rows = [(r["type"], r["ko"], r["en"]) for r in csv.DictReader(f, delimiter="\t")]
    return [{"type": t, "ko": k, "en": e} for t, k, e in rows + load_eval_queries()]


def clean(raw: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    s = s.split("\n")[0].strip()
    s = re.sub(r"^(English|Translation)\s*:\s*", "", s, flags=re.I).strip()
    return re.sub(r'^["\'“‘]|["\'”’]$', "", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(MODELS))
    args = ap.parse_args()
    model_id, need_gb = MODELS[args.tag]

    import torch
    from vllm import LLM, SamplingParams

    # 다른 사용자의 작업이 GPU를 쓰고 있으면 기동 중에 vLLM이 실패하므로 먼저 확인한다.
    free, total = torch.cuda.mem_get_info()
    if free < (need_gb + 1) * 2**30:
        raise SystemExit(f"GPU 여유 {free / 2**30:.1f}GB < 필요 {need_gb}GB — 다른 작업이 끝난 뒤 다시")
    util = need_gb * 2**30 / total

    items = load_items()
    # 이미지 입력을 받는 모델(Gemma 3, Qwen2.5-VL)은 비전 쪽 메모리를 잡지 않게 끈다
    mm = {"limit_mm_per_prompt": {"image": 0}} if args.tag in ("gemma3_27b", "gemma3_4b", "qwen25vl_7b") else {}
    llm = LLM(model=model_id, gpu_memory_utilization=util, max_model_len=1024,
              enforce_eager=True, **mm)
    msgs = [[{"role": "user", "content": PROMPT.format(ko=it["ko"])}] for it in items]
    outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=96))
    for it, o in zip(items, outs):
        it["translation"] = clean(o.outputs[0].text)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.tag}.json"
    json.dump({"model": model_id, "prompt": PROMPT, "items": items}, open(path, "w"),
              ensure_ascii=False, indent=1)
    print(f"{len(items)}개 → {path}")
    for it in items[:8]:
        print(f"  {it['ko']} → {it['translation']}")


if __name__ == "__main__":
    main()
