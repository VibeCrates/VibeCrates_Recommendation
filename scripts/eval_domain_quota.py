"""도메인 쿼터 후처리 실험 (3-1). 재학습도 재인덱싱도 없다.

무엇을 바꾸는가:
  통합검색(domain_filter=all)에서 도메인별 인덱스를 합쳐 정렬하는 지점에, 도메인당
  상위 cap개만 남기는 상한을 건다(eval_lang.search의 domain_cap). 다른 것은 전부 같다.

**고정 비율이 아니라 상한을 먼저 시험한다.** 고정 비율(영화 3/음악 3/책 4)은 쏠림을
확실히 없애지만 그 쿼리에 정말 책만 어울릴 때도 음악을 끼워 넣는다. 상한은 그 손해가 작다.

무엇을 보는가 — 판정 평균만 보면 안 된다:
  (1) 판정 평균 — judge_eval_relevance.py가 읽을 CSV를 낸다.
  (2) 쏠림 지표 — 판정기 없이 직접 센다. 쿼리별 최대 도메인 비중과 도메인 분포.
  (3) 밀려난 아이템 — 상한 때문에 빠진 것과 새로 들어온 것을 함께 남긴다.

판정 평균이 내려가는데 화면이 더 좋아 보이면, 그것은 우리 지표가 서비스 요구(한 크레이트에
세 도메인이 섞임)를 재지 않는다는 뜻이고 평가 설계를 고쳐야 한다는 신호다.

사용:
  python scripts/eval_domain_quota.py --caps 5 \
      --model-path models/best_stage2_qlora_off.pt --index-dir indexes_qlora_off
"""
import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.eval_lang import (QUERIES, STYLE_NAMES, TOP_K, load_indexes,
                               load_model, search)


# 쿼리가 도메인을 이름으로 지목하면 상한을 걸지 않는다.
# 9/9 실측: direct 쿼리 10개가 전부 도메인 단어를 담고 있고 나머지 30개는 하나도 없다.
# 상한을 무조건 걸면 direct에서 유의하게 나빠진다(쌍대 17.6%, p=0.0002) — 사용자가
# "우주 배경 액션 **영화**"를 찾는데 음악을 끼워 넣기 때문이다.
DOMAIN_WORD = re.compile(
    r"\b(movie|movies|film|films|cinema|music|song|songs|track|tracks|album|albums|"
    r"book|books|novel|novels|fiction|nonfiction|memoir|drama|piano|guitar|rock|jazz|"
    r"soundtrack|documentary)\b", re.I)


def names_domain(query: str) -> bool:
    return bool(DOMAIN_WORD.search(query))


def run(model, indexes, cap, top_k, guard=False):
    rows = []
    for c, p, lang, q in QUERIES:
        if lang != "en":          # 쿼터는 통합검색 구성 문제이므로 영어로 먼저 본다
            continue
        use_cap = None if (guard and cap is not None and names_domain(q)) else cap
        for r in search(model, indexes, q, None, top_k, domain_cap=use_cap):
            rows.append({
                "query_id": f"{c}{p}_EN", "style": STYLE_NAMES[c], "pair_id": p,
                "lang": "en", "query": q, "domain_filter": "all", **r,
            })
    return pd.DataFrame(rows)


def concentration(df: pd.DataFrame) -> dict:
    """판정기 없이 세는 쏠림 지표."""
    per_query = []
    for _, g in df.groupby("query_id"):
        cnt = Counter(g.result_domain)
        per_query.append({
            "top_share": max(cnt.values()) / len(g),      # 한 도메인이 차지한 최대 비중
            "n_domains": len(cnt),                        # 등장한 도메인 수
            "single": int(len(cnt) == 1),                 # 한 도메인이 통째로 먹은 쿼리
        })
    pq = pd.DataFrame(per_query)
    out = {"쿼리별 최대도메인 비중": round(pq.top_share.mean(), 3),
           "평균 도메인 수": round(pq.n_domains.mean(), 2),
           "한 도메인 독식 쿼리": int(pq.single.sum())}
    share = df.result_domain.value_counts(normalize=True)
    for d in ("movie", "music", "book"):
        out[f"{d} 비중"] = round(float(share.get(d, 0.0)), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", nargs="+", type=int, default=[5])
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--model-path", default="models/best_stage2_qlora_off.pt")
    ap.add_argument("--index-dir", default="indexes_qlora_off")
    ap.add_argument("--out-prefix", default="experiments/eval_quota")
    args = ap.parse_args()

    model = load_model(args.model_path)
    indexes = load_indexes(args.index_dir)

    runs = {"base": run(model, indexes, None, args.top_k)}
    for cap in args.caps:
        runs[f"cap{cap}"] = run(model, indexes, cap, args.top_k)
        runs[f"cap{cap}_guard"] = run(model, indexes, cap, args.top_k, guard=True)

    print("\n=== 쏠림 지표 (판정기 없이 직접 센다) ===")
    stat = pd.DataFrame({k: concentration(v) for k, v in runs.items()}).T
    print(stat.to_string())

    base = runs["base"]
    bset = {(r.query_id, str(r.item_id)) for r in base.itertuples()}
    for name, df in runs.items():
        if name == "base":
            continue
        cur = {(r.query_id, str(r.item_id)) for r in df.itertuples()}
        print(f"\n=== {name}: base 대비 교체 {len(bset - cur)}건 / 400 "
              f"({100*len(bset - cur)/len(bset):.1f}%) ===")
        out_dom = Counter(r.result_domain for r in base.itertuples()
                          if (r.query_id, str(r.item_id)) in bset - cur)
        in_dom = Counter(r.result_domain for r in df.itertuples()
                         if (r.query_id, str(r.item_id)) in cur - bset)
        print(f"  밀려난 것: {dict(out_dom)}")
        print(f"  들어온 것: {dict(in_dom)}")

    for name, df in runs.items():
        out = f"{args.out_prefix}_{name}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[{name}] {len(df):,}행 → {out}")
    stat.to_csv(f"{args.out_prefix}_concentration.csv")


if __name__ == "__main__":
    main()
