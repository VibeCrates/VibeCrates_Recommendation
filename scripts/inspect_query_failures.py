"""
DSV 검증에 반복 실패하는 아이템이 어떤 것들인지 프로파일링한다.

왜 필요한가:
  movie 쿼리를 재실행했더니 잔여 6,377건 중 유효 DSV가 1,182건(18.5%)뿐이었다.
  이 6,377건은 이전 실행에서도 실패해 캐시에 안 들어간 것들이다 — 즉 "몇 번을 돌려도
  안 되는 층"이 있다. 최종 커버리지 34,412/40,109(85.8%)이고 나머지는 쿼리 없이 남아
  prepare_dataset에서 드롭된다. 그 5,697건이 무작위인지 특정 성격의 아이템인지에 따라
  대응이 갈린다 — 무작위면 무시해도 되고, 특정 층(예: 비영어 제목, 소스 없는 항목)이면
  그 층 전체가 학습에서 통째로 빠지는 편향이 된다.

두 가지 모드:
  (기본) 프로파일 — CPU만. 실패군과 성공군을 특성별로 비교하고 층별 실패율을 낸다.
  --probe N — GPU. 실패 아이템 N건을 다시 생성해 **원본 출력**과 거절 사유를 덤프한다.
      캐시에는 검증 통과분만 저장되므로, 모델이 실제로 뭘 뱉었는지는 이 방법으로만 볼 수
      있다. vLLM 엔진을 새로 띄우므로 다른 GPU 작업이 없을 때 실행할 것.

사용:
  python scripts/inspect_query_failures.py --domain movie
  python scripts/inspect_query_failures.py --domain movie --probe 64   # GPU 여유 있을 때
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from scripts.generate_queries import (
    DOMAIN_CONFIGS, PROMPT_TEMPLATE, build_synopsis, load_image, validate_dsv,
)

CACHE_DIR = "data/cache"
LOCAL_IMAGE_DIRS = {"movie": "data/images/movie", "music": "data/images/music",
                    "book": "data/images/book"}


def why_invalid(raw: str) -> str:
    """validate_dsv가 거절한 이유를 분류한다 (validate_dsv의 판정 순서를 그대로 따른다)."""
    text = str(raw).strip()
    if not text:
        return "빈 출력"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "빈 출력"
    if all(l.startswith("[") for l in lines):
        return "라벨 줄만 (예: [Output])"

    reasons = []
    for line in lines:
        if line.startswith("["):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 1:
            reasons.append("구분자(|) 없음")
        elif len(parts) != 3:
            reasons.append(f"파트 {len(parts)}개 (3개여야 함)")
        elif not all(parts):
            reasons.append("빈 파트 포함")
    return reasons[0] if reasons else "알 수 없음"


def profile(domain: str, args) -> pd.DataFrame:
    cfg = DOMAIN_CONFIGS[domain]
    id_col = cfg["id_col"]
    df = pd.read_csv(cfg["csv"], low_memory=False)
    df[id_col] = df[id_col].astype(str)

    cache_path = f"{CACHE_DIR}/query_cache_{domain}.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    df["_failed"] = ~df[id_col].isin(cache.keys())

    n_fail = int(df["_failed"].sum())
    print(f"\n{'='*72}\n[{domain}] {len(df):,}행 · 쿼리 실패 {n_fail:,}건 ({n_fail/len(df):.1%})\n{'='*72}")
    if not n_fail:
        return df

    # ── 특성별 비교 ──────────────────────────────────────────────────────────
    synth = df.get("description_synth", pd.Series("", index=df.index)).fillna("").astype(str)
    title_col = {"movie": "Title", "music": "name", "book": "title"}[domain]
    titles = df[title_col].fillna("").astype(str)

    df["_synth_len"] = synth.str.len()
    df["_has_synth"] = synth.str.strip().ne("")
    df["_title_nonascii"] = titles.map(lambda s: sum(ord(c) > 127 for c in s) / max(len(s), 1))
    df["_has_url"] = df[cfg["image_col"]].astype(str).str.startswith("http")
    df["_has_local_img"] = [
        os.path.exists(os.path.join(LOCAL_IMAGE_DIRS[domain], f"{i}.jpg")) for i in df[id_col]
    ]
    df["_synopsis_len"] = [len(build_synopsis(domain, r)) for _, r in df.iterrows()]

    feats = ["_has_synth", "_synth_len", "_has_url", "_has_local_img",
             "_title_nonascii", "_synopsis_len"]
    ok, bad = df[~df["_failed"]], df[df["_failed"]]
    print(f"\n특성 비교 (성공 {len(ok):,} vs 실패 {len(bad):,})")
    print(f"    {'특성':18s} {'성공군':>10s} {'실패군':>10s} {'배율':>8s}")
    for f in feats:
        a, b = ok[f].mean(), bad[f].mean()
        ratio = (b / a) if a else float("nan")
        mark = "  ←" if abs(ratio - 1) > 0.25 else ""
        print(f"    {f[1:]:18s} {a:>10.3f} {b:>10.3f} {ratio:>8.2f}{mark}")

    # ── 층별 실패율 ──────────────────────────────────────────────────────────
    keys = []
    if "description_synth_basis" in df.columns:
        keys.append("description_synth_basis")
    if domain == "book":
        df["_src"] = df[id_col].str[:3].str.rstrip("_")
        keys.append("_src")
    for key in keys:
        print(f"\n층별 실패율 — {key}")
        g = df.groupby(key)["_failed"].agg(["size", "sum"])
        g["실패율"] = g["sum"] / g["size"]
        for name, row in g.sort_values("실패율", ascending=False).iterrows():
            print(f"    {str(name):24s} {int(row['size']):>8,}건  실패 {int(row['sum']):>7,}  {row['실패율']:.1%}")

    # ── 제목 비영어 구간별 ───────────────────────────────────────────────────
    bins = [0, 0.001, 0.2, 0.5, 1.01]
    labels = ["ASCII만", "~20%", "20~50%", "50%+"]
    df["_na_bin"] = pd.cut(df["_title_nonascii"], bins=bins, labels=labels, right=False)
    print("\n제목의 비ASCII 비율별 실패율")
    g = df.groupby("_na_bin", observed=True)["_failed"].agg(["size", "sum"])
    for name, row in g.iterrows():
        if row["size"]:
            print(f"    {str(name):12s} {int(row['size']):>8,}건  실패 {int(row['sum']):>7,}  "
                  f"{row['sum']/row['size']:.1%}")

    # ── 실패 샘플 덤프 ───────────────────────────────────────────────────────
    dump = f"{CACHE_DIR}/query_failures_{domain}.txt"
    with open(dump, "w") as f:
        for _, r in bad.head(args.dump).iterrows():
            f.write(f"\n{'-'*72}\n[{r[id_col]}] {r[title_col]}\n")
            f.write(f"이미지: url={r['_has_url']} local={r['_has_local_img']}\n")
            f.write(f"— Qwen에게 들어간 텍스트 —\n{build_synopsis(domain, r)}\n")
    print(f"\n실패 샘플 {min(args.dump, n_fail)}건 → {dump}")
    return df


def probe(domain: str, df: pd.DataFrame, n: int) -> None:
    """실패 아이템을 다시 생성해 원본 출력과 거절 사유를 본다 (GPU)."""
    from collections import Counter
    from scripts.vllm_runner import VLLMRunner

    cfg = DOMAIN_CONFIGS[domain]
    id_col = cfg["id_col"]
    bad = df[df["_failed"]].head(n)
    print(f"\n{'='*72}\nprobe: 실패 {len(bad)}건 재생성 (vLLM)\n{'='*72}")

    items = []
    for _, r in bad.iterrows():
        try:
            img = load_image(domain, str(r[id_col]), str(r[cfg["image_col"]])) \
                if cfg["has_image"](r) else None
        except Exception:
            img = None
        items.append((PROMPT_TEMPLATE.format(synopsis=build_synopsis(domain, r),
                                             role=cfg["role"]), img))

    runner = VLLMRunner(cfg.get("model_id", "Qwen/Qwen2.5-VL-7B-Instruct"), max_new_tokens=64)
    outputs = runner.generate(items)

    reasons, recovered = Counter(), 0
    dump = f"{CACHE_DIR}/query_failures_{domain}_probe.txt"
    with open(dump, "w") as f:
        for (_, r), raw in zip(bad.iterrows(), outputs):
            dsv = validate_dsv(raw)
            reason = "재시도 성공" if dsv else why_invalid(raw)
            reasons[reason] += 1
            recovered += bool(dsv)
            f.write(f"\n{'-'*72}\n[{r[id_col]}] {r[{'movie':'Title','music':'name','book':'title'}[domain]]}\n")
            f.write(f"판정: {reason}\n원본 출력: {raw!r}\n")

    print(f"\n거절 사유 분포 ({len(bad)}건)")
    for reason, c in reasons.most_common():
        print(f"    {reason:28s} {c:>5,}  {c/len(bad):.1%}")
    print(f"\n재시도로 살아난 건 {recovered:,}건 ({recovered/len(bad):.1%})")
    print(f"원본 출력 덤프 → {dump}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", nargs="+", default=["movie"], choices=list(DOMAIN_CONFIGS))
    ap.add_argument("--dump", type=int, default=30, help="실패 샘플 덤프 건수")
    ap.add_argument("--probe", type=int, default=0,
                    help="GPU로 재생성해 원본 출력을 볼 건수 (0=안 함)")
    args = ap.parse_args()

    for domain in args.domain:
        df = profile(domain, args)
        if args.probe and df["_failed"].any():
            probe(domain, df, args.probe)


if __name__ == "__main__":
    main()
