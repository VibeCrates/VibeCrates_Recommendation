"""한→영 검색어 번역 사전을 만든다 (2026-10-12).

왜 필요한가:
  서빙 번역기 opus-mt-ko-en은 백엔드가 실제로 보내는 짧은 검색어에서 무너진다. 로그의 한국어
  검색어 530종 중 33.0%가 핵심어를 잃었다 — 시집 → "Indulgent", 보사노바 → "Bosanoba",
  이별 → "DIVORCE", 추리소설 → "The Story of Reason". 같은 530종을 Gemma 27B로 옮기면 0.9%다
  (experiments/translation_ko530_*.csv). 하지만 27B는 서빙 중인 Mac에서 돌릴 수 없다.

  그 검색어들은 장르·분위기·주제·상황 단어 약 70개의 조합이었다. 단어 수가 한정돼 있으니
  미리 옮겨 두면 된다. 서버는 검색어를 이 사전의 단어로 끝까지 쪼갤 수 있으면 사전 번역을
  이어 붙여 쓰고, 못 쪼개면 지금처럼 opus-mt로 옮긴다(src/api/translation.py).

어휘와 번역:
  data/translation/ko_vocab.tsv — 한국어 <TAB> 분류 <TAB> 고정 번역(선택).
  고정 번역은 사람이 검토해 정한 값이고, 있으면 그것을 쓴다. 없으면 Gemma 번역을 쓴다.
  Gemma는 모든 항목을 옮겨 고정 번역과 대조하는 데도 쓴다 — 둘이 다르면 검토 목록에 남긴다.

사용:
  # 1) beast: 어휘 전체를 Gemma로 번역해 캐시에 저장
  python scripts/build_translation_dict.py --translate
  # 2) 어디서나: 캐시 + 고정 번역으로 사전을 쓴다
  python scripts/build_translation_dict.py
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VOCAB = "data/translation/ko_vocab.tsv"
GEMMA_CACHE = "data/translation/ko_vocab_gemma.json"
OUT = "src/api/ko_en_dict.json"
REVIEW = "experiments/translation_dict_review.csv"
GEMMA = "RedHatAI/gemma-3-27b-it-quantized.w4a16"

# 로그 530종 비교에 쓴 프롬프트와 같다. 첫 시도에서 "숲" → "forest films"처럼 원문에 없는
# 도메인 단어를 붙여서, 붙이지 말라는 규칙과 맨 주제어 예시(외로움)를 넣었다.
PROMPT = (
    "Translate the Korean search query below into natural English.\n"
    "Users type these into a search box to find films, songs, or books by mood, genre, or topic. "
    "Many are only one or two words.\n"
    "Rules:\n"
    "- Keep it a short search phrase with the same meaning. Do not turn it into a sentence.\n"
    "- Never add words the Korean does not have. In particular, do not add film, song, music, "
    "book, novel or similar unless the Korean names that kind of work; a bare topic stays a bare "
    "topic.\n"
    "- If the Korean does name a kind of work (영화, 노래, 음악, 앨범, 책, 소설, 시집, 에세이 ...), "
    "keep it.\n"
    "- Translate meaning, not sound. Use the usual English term for genres.\n"
    "- Keep numbers as they are.\n"
    "- Output only the English phrase. No quotes, no explanation.\n\n"
    "Examples:\n"
    "노을 지는 바닷가 → seaside at sunset\n"
    "연애소설 → romance novels\n"
    "출근길 플레이리스트 → morning commute playlist\n"
    "몽환적인 신스팝 → dreamy synth-pop\n"
    "한겨울 산장 → mountain cabin in deep winter\n"
    "외로움 → loneliness\n"
    "혼자 걷는 밤 → walking alone at night\n\n"
    "Korean: {q}\nEnglish:"
)


def read_vocab(path: str = VOCAB) -> list[dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        ko, cat = parts[0].strip(), parts[1].strip()
        fixed = parts[2].strip() if len(parts) > 2 else None
        # 군더더기(filler)는 번역하지 않고 지운다 — "비 오는 날 듣기 좋은 노래" → "rainy day songs"
        if cat == "filler":
            fixed = ""
        rows.append({"ko": ko, "category": cat, "fixed": fixed})
    keys = [r["ko"] for r in rows]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        sys.exit(f"어휘 중복: {sorted(dup)}")
    return rows


def clean(s: str) -> str:
    s = str(s).strip().split("\n")[0]
    if "→" in s:
        s = s.split("→")[-1]
    return s.strip().strip("\"'“”‘’ ").strip()


def translate(rows: list[dict]) -> None:
    from scripts.vllm_runner import VLLMRunner
    cache = json.load(open(GEMMA_CACHE)) if os.path.exists(GEMMA_CACHE) else {}
    todo = [r["ko"] for r in rows if r["ko"] not in cache]
    print(f"번역 대상 {len(todo)} (캐시 {len(rows) - len(todo)})")
    if todo:
        runner = VLLMRunner(GEMMA, max_new_tokens=32, repetition_penalty=1.0,
                            gpu_memory_utilization=0.8)
        outs = runner.generate([(PROMPT.format(q=q), None) for q in todo])
        cache.update({q: clean(o) for q, o in zip(todo, outs)})
        with open(GEMMA_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"→ {GEMMA_CACHE}")


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def build(rows: list[dict]) -> None:
    import pandas as pd
    gemma = json.load(open(GEMMA_CACHE)) if os.path.exists(GEMMA_CACHE) else {}
    entries, review = {}, []
    for r in rows:
        g = gemma.get(r["ko"])
        en = r["fixed"] if r["fixed"] is not None else g
        if en is None:
            sys.exit(f"번역 없음: {r['ko']} — beast에서 --translate를 먼저 돌린다")
        if re.search(r"[가-힣]", en):
            sys.exit(f"번역에 한글이 남음: {r['ko']} → {en}")
        entries[norm(r["ko"])] = en
        if r["category"] != "filler" and g is not None and norm(g) != norm(en):
            review.append({"ko": r["ko"], "category": r["category"], "used": en, "gemma": g})
    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": VOCAB,
        "note": "키는 소문자·공백 정규화한 한국어 단위, 값은 영어. 빈 문자열은 지우는 군더더기다.",
        "entries": dict(sorted(entries.items())),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    pd.DataFrame(review).to_csv(REVIEW, index=False)
    fixed = sum(r["fixed"] is not None for r in rows)
    print(f"사전 {len(entries)}개 → {OUT} (고정 번역 {fixed}, Gemma {len(entries) - fixed})")
    print(f"고정 번역과 Gemma가 다른 항목 {len(review)}개 → {REVIEW}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translate", action="store_true", help="어휘를 Gemma로 번역해 캐시에 저장 (GPU)")
    args = ap.parse_args()
    rows = read_vocab()
    if args.translate:
        translate(rows)
    else:
        build(rows)


if __name__ == "__main__":
    main()
