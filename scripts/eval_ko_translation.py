"""번역기별로 한국어 검색 품질을 잰다 (재학습 없음, 인덱스 그대로).

무엇을 비교하는가 — 검색에 넣는 문자열만 바꾸고 나머지는 전부 고정한다.

  ko_raw : 한국어 원문을 그대로 인코딩 (지금 experiments/eval_lang_ext_*.csv의 ko 절반)
  opus   : opus-mt-ko-en 번역 (현재 서빙 경로, src/api/translation.py)
  qwen   : Qwen 번역 (교체 후보)

**판정용 쿼리는 세 경우 모두 사람이 쓴 영어 원문으로 고정한다.** 한국어로 판정하면
판정기 자체의 한국어 열위가 섞인다 — 9/8 대조 실험에서 그 몫이 0.067(전체 격차의
10.5%)로 측정됐다. 영어 원문으로 고정하면 남는 차이가 전부 검색 품질이다.

그래서 출력 CSV는 `query`에 영어 원문을, `query_used`에 실제로 인코딩한 문자열을 담고
`lang`을 en으로 적는다. judge_eval_relevance.py가 수정 없이 읽고, 같은 (아이템, 영어
쿼리) 쌍은 캐시를 공유한다.

사용:
  python scripts/eval_ko_translation.py                  # 세 모드 전부
  python scripts/eval_ko_translation.py --modes qwen
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn.functional as F

from scripts.eval_lang import (DOMAINS, QUERIES, STYLE_NAMES, TOP_K,
                               load_indexes, load_model, search)

TRANS = "data/cache/eval_query_translations.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["ko_raw", "opus", "qwen"])
    ap.add_argument("--translations", default=TRANS)
    ap.add_argument("--model-path", default=os.environ.get("EVAL_MODEL_PATH", "models/trained_model.pt"))
    ap.add_argument("--index-dir", default=os.environ.get("EVAL_INDEX_DIR", "indexes"))
    ap.add_argument("--out-prefix", default="experiments/eval_kotrans")
    args = ap.parse_args()

    trans = json.load(open(args.translations)) if os.path.exists(args.translations) else {}
    ko = [(c, p, q) for c, p, l, q in QUERIES if l == "ko"]
    en = {(c, p): q for c, p, l, q in QUERIES if l == "en"}

    model = load_model(args.model_path)
    indexes = load_indexes(args.index_dir)

    # ── 쿼리 벡터 코사인: 각 모드가 영어 원문 벡터에 얼마나 가까운가 ──────────────
    # 판정 이전에 이 값만 봐도 번역이 작동하는지 보인다. 세션 20이 ko_raw 0.058 /
    # opus 0.780으로 잰 것과 같은 계산이다.
    print("\n=== 쿼리 벡터 코사인 (영어 원문 대비) ===")
    cos_rows = []
    for c, p, q_ko in ko:
        qid = f"{c}{p}_KO"
        texts = {"ko_raw": q_ko, "opus": trans.get("opus", {}).get(qid),
                 "qwen": trans.get("qwen", {}).get(qid)}
        z_en = F.normalize(model.encode_query([en[(c, p)]]), p=2, dim=1)
        row = {"style": STYLE_NAMES[c], "pair_id": p, "query_ko": q_ko}
        for m, t in texts.items():
            if not t:
                continue
            z = F.normalize(model.encode_query([t]), p=2, dim=1)
            row[m] = float((z_en @ z.T).squeeze())
        cos_rows.append(row)
    cos = pd.DataFrame(cos_rows)
    present = [m for m in ("ko_raw", "opus", "qwen") if m in cos.columns]
    print(cos.groupby("style")[present].mean().round(3).to_string())
    print("  전체:", {m: round(cos[m].mean(), 3) for m in present})
    cos.to_csv(f"{args.out_prefix}_cosine.csv", index=False)

    # ── 검색 ─────────────────────────────────────────────────────────────────
    for mode in args.modes:
        rows = []
        for c, p, q_ko in ko:
            qid = f"{c}{p}_KO"
            used = q_ko if mode == "ko_raw" else trans.get(mode, {}).get(qid)
            if not used:
                print(f"  [warn] {mode}: {qid} 번역 없음 — 건너뜀")
                continue
            for domain_filter in DOMAINS:
                df_arg = None if domain_filter == "all" else domain_filter
                if df_arg and df_arg not in indexes:
                    continue
                for r in search(model, indexes, used, df_arg, TOP_K):
                    rows.append({
                        "query_id": f"{c}{p}_KO_{mode}", "style": STYLE_NAMES[c],
                        "pair_id": p, "lang": "en",          # 판정은 영어로 한다
                        "query": en[(c, p)],                 # 판정기가 보는 쿼리
                        "query_used": used,                  # 실제로 인코딩한 문자열
                        "domain_filter": domain_filter, **r,
                    })
        out = f"{args.out_prefix}_{mode}.csv"
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[{mode}] {len(rows):,}행 → {out}")


if __name__ == "__main__":
    main()
