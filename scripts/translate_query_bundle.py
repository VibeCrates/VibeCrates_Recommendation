"""학습 쿼리 벡터 번들을 정리하고 한국어 쿼리 열을 붙인다 (2026-10-09 사용자 요청).

무엇을 하는가:
  1. 타 언어가 섞인 쿼리를 뺀다. 학습 쿼리는 Qwen에게 "영어로만"을 지시해 만들었지만
     약 1%에 중국어·일본어·아랍어 등이 섞였다(`Coca-Cola瓶的奇迹时刻`). 라틴 문자가 아닌
     글자가 하나라도 있으면 뺀다. 악센트 라틴 문자(é, ô)와 둥근 따옴표·대시 같은 기호는
     정상 영어에서도 쓰이므로 남긴다.
  2. 남은 영어 쿼리를 LLM으로 한국어로 옮겨 `query_ko` 열을 붙인다.
     **벡터는 바꾸지 않는다** — 모델은 영어로만 학습됐고, 벡터는 영어 쿼리로 만든 것이다.
     `query_ko`는 표시용 번역이다. 같은 한국어를 API에 넣으면 API가 다시 영어로 번역해
     인코딩하므로 번들 벡터와 조금 다른 벡터가 나온다.

번역 결과는 캐시(`data/cache/query_ko_<model>.json`)에 쌓아, 끊겨도 이어서 돈다.

사용 (beast):
  # 시험 번역 (모델 비교용)
  python scripts/translate_query_bundle.py --src dist/query_vectors_20261009 \\
      --model-id RedHatAI/gemma-3-27b-it-quantized.w4a16 --gpu-util 0.8 --pilot 200
  # 전체
  python scripts/translate_query_bundle.py --src dist/query_vectors_20261009 \\
      --out dist/query_vectors_20261009_ko --model-id ... --gpu-util 0.8
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

DOMAINS = ("movie", "music", "book")
CACHE_DIR = "data/cache"

PROMPT = (
    "Translate the English search phrase below into natural Korean.\n"
    "It is a short, evocative phrase someone might type to find a film, song, or book by mood.\n"
    "Rules:\n"
    "- Keep it a short phrase (not a full sentence), with the same mood and imagery.\n"
    "- Keep proper nouns (people, places, brands) in their usual Korean form or as-is.\n"
    "- Output only the Korean phrase. No quotes, no explanation, no romanization.\n\n"
    "Examples:\n"
    "fragments of a blue dawn → 푸른 새벽의 조각들\n"
    "rainy day vinyl bar → 비 오는 날의 LP 바\n"
    "at the edge of existence → 존재의 끝자락에서\n"
    "Bobby Fischer’s inner storm → 바비 피셔의 내면의 폭풍\n\n"
    "English: {q}\nKorean:"
)

# 1차 번역에서 0.065%(241/372,667)가 한자(陰鬱)·러시아어(открытое небо, "open"이 든 쿼리)를
# 섞었다. 그것만 더 엄격한 지시로 다시 돌린다(--retry-bad).
PROMPT_STRICT = PROMPT.replace(
    "- Output only the Korean phrase.",
    "- Write in Korean Hangul only. Never use Chinese characters (Hanja), Russian, or any other "
    "script; Latin letters are allowed only for brand names.\n- Output only the Korean phrase.")
# 엄격한 지시로도 138개가 남았다 — 특정 단어마다 굳은 버릇이 있다(grim→陰鬱, whispers→шепот,
# awakening→пробуждение, street corner→街角, bloom→绽花, "open ..."→러시아어). 그 단어들의
# 한국어 예시를 직접 보여 준다.
PROMPT_STRICT = PROMPT_STRICT.replace(
    "Bobby Fischer’s inner storm → 바비 피셔의 내면의 폭풍\n",
    "Bobby Fischer’s inner storm → 바비 피셔의 내면의 폭풍\n"
    "open sky horizon → 탁 트인 하늘의 지평선\n"
    "open road journey → 끝없이 열린 길 위의 여정\n"
    "grim shadows of silence → 침묵의 음울한 그림자\n"
    "night forest whispers → 밤 숲의 속삭임\n"
    "fiery passion awakening → 깨어나는 불타는 열정\n"
    "lonely city street corner → 쓸쓸한 도시의 길모퉁이\n"
    "wild berries bloom → 피어나는 산딸기\n"
    "zen meditation path → 선 명상의 길\n")

HANGUL = re.compile(r"[가-힣]")
OTHER_SCRIPT = re.compile(r"[぀-ヿ㐀-鿿؀-ۿЀ-ӿ฀-๿]")


def is_latin_only(s: str) -> bool:
    """글자는 모두 라틴(악센트 포함)이어야 한다. 숫자·기호는 상관없다."""
    for c in s:
        if c.isalpha() and ord(c) > 127:
            try:
                if not unicodedata.name(c).startswith("LATIN"):
                    return False
            except ValueError:
                return False
    return True


def clean_ko(raw: str) -> str:
    """첫 줄만, 따옴표·화살표 접두사 제거.

    - 모델이 "открытое небо → 열린 하늘"처럼 화살표 뒤에 한국어를 다시 쓰면 마지막 부분을 쓴다.
    - 괄호 속 한자 병기(선(禪), 만(灣), 비가(悲歌))는 지운다 — 한국어에서 흔한 표기지만 한자가
      남으면 타 문자로 걸리고, 지워도 뜻은 그대로다.
    """
    s = str(raw).strip().split("\n")[0].strip()
    if "→" in s:
        s = s.split("→")[-1]
    s = re.sub(r"^(Korean\s*:|한국어\s*:)\s*", "", s.strip()).strip()
    s = re.sub(r"\s*\([^)]*[\u3400-\u9fff][^)]*\)", "", s)
    return s.strip("\"'“”‘’ ").strip()


def check_ko(s: str) -> str | None:
    """번역 결과의 문제를 한 단어로. 문제없으면 None."""
    if not s:
        return "empty"
    if not HANGUL.search(s):
        return "no_hangul"
    if OTHER_SCRIPT.search(s):
        return "other_script"
    if len(s) > 60:
        return "too_long"
    return None


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--gpu-util", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--pilot", type=int, default=0, help="무작위 N개만 번역해 품질·속도를 본다")
    ap.add_argument("--retry-bad", action="store_true",
                    help="캐시에서 한자·타 문자·한글 없음으로 걸린 번역만 엄격한 지시로 다시 번역")
    args = ap.parse_args()

    frames = {d: pd.read_csv(f"{args.src}/{d}_query_vectors.csv") for d in DOMAINS}
    uniq = sorted(set().union(*(set(f["query"]) for f in frames.values())))
    keep = [q for q in uniq if is_latin_only(q)]
    dropped = sorted(set(uniq) - set(keep))
    print(f"고유 쿼리 {len(uniq):,} → 라틴 문자만 {len(keep):,} (제외 {len(dropped):,})")

    tag = re.sub(r"[^A-Za-z0-9.-]+", "-", args.model_id)
    cache_path = os.path.join(CACHE_DIR, f"query_ko_{tag}.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    targets = keep
    if args.pilot:
        import random
        targets = random.Random(20261009).sample(keep, args.pilot)
    retry = args.retry_bad
    if retry:
        todo = [q for q in targets if check_ko(cache.get(q, "")) in ("other_script", "no_hangul")]
    else:
        todo = [q for q in targets if q not in cache]
    print(f"번역 대상 {len(targets):,} (캐시 {len(targets) - len(todo):,}, 새로 {len(todo):,})")

    if todo:
        from scripts.vllm_runner import VLLMRunner
        runner = VLLMRunner(args.model_id, max_new_tokens=48, repetition_penalty=1.0,
                            gpu_memory_utilization=args.gpu_util)
        os.makedirs(CACHE_DIR, exist_ok=True)
        t0 = time.time()
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            prompt = PROMPT_STRICT if retry else PROMPT
            outs = runner.generate([(prompt.format(q=q), None) for q in chunk])
            for q, raw in zip(chunk, outs):
                cache[q] = clean_ko(raw)
            json.dump(cache, open(cache_path, "w"), ensure_ascii=False)
            done = i + len(chunk)
            print(f"  {done:,}/{len(todo):,}  {time.time() - t0:.0f}s "
                  f"({done / max(time.time() - t0, 1e-9):.1f}개/초)", flush=True)

    probs = pd.Series([check_ko(cache.get(q, "")) for q in targets]).value_counts(dropna=False)
    print("번역 점검:", probs.to_dict())
    if args.pilot:
        for q in targets[:40]:
            print(f"  {q}  →  {cache.get(q)}")
        return

    # ── 새 번들 쓰기 ──────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    keep_set = set(keep)
    man = json.load(open(f"{args.src}/manifest.json"))
    man["created_at"] = datetime.now(timezone.utc).isoformat()
    man["derived_from"] = os.path.basename(os.path.normpath(args.src))
    man["removed_non_latin"] = len(dropped)
    man["translation_model"] = args.model_id
    man["translation_issues"] = {str(k): int(v) for k, v in probs.items()}
    man["note"] = (
        "각 {domain}_query_vectors.csv는 열 네 개 — query, query_ko, vector_b64, model_version. "
        "vector_b64는 영어 query로 만든 벡터다((768,) float32, L2 정규화, "
        "`np.frombuffer(base64.b64decode(v), dtype='<f4')`). query_ko는 표시용 번역으로, 벡터는 "
        "query_ko로 만든 것이 아니다. 라틴 문자가 아닌 글자가 섞인 쿼리는 제외했다.")
    man["unique_queries_total"] = len(keep)
    for d, f in frames.items():
        f = f[f["query"].isin(keep_set)].copy()
        # 타 문자·한글 없음이 끝까지 남은 번역은 비운다 — 섞인 채로 넘기는 것보다 "번역 없음"이
        # 안전하다. 원문이 깨진 too_long은 번역 자체는 한국어라 그대로 둔다.
        ko = f["query"].map(lambda q: cache.get(q, "")
                            if check_ko(cache.get(q, "")) in (None, "too_long") else "")
        f.insert(1, "query_ko", ko)
        path = f"{args.out}/{d}_query_vectors.csv"
        f.to_csv(path, index=False, encoding="utf-8")
        man["domains"][d] = {"count": len(f), "file": os.path.basename(path),
                             "sha256": sha256(path), "bytes": os.path.getsize(path)}
        print(f"  {d}: {len(f):,}행 → {path}")
    with open(f"{args.out}/manifest.json", "w") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=2)
    pd.Series(dropped, name="query").to_csv(f"{args.out}/removed_non_latin.csv", index=False)
    print(f"manifest → {args.out}/manifest.json, 제외 목록 → removed_non_latin.csv")


if __name__ == "__main__":
    main()
