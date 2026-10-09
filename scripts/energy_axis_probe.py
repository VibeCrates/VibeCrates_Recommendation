"""energy 축 진단 — 검색 시점 재가중(worklist 3-3)을 걸기 전에 쿼리 쪽이 준비됐는지 본다.

배경 (worklog 2-6):
  음악 아이템 벡터에서 energy를 선형으로 되찾으면 R² 0.598이다 — 정보는 있다. 그러나 그
  방향으로 사영한 폭이 표준편차 0.007뿐이라 검색 점수에 거의 영향을 못 준다. 그래서
  "그 방향을 넓히면 된다"가 3-3의 가설이다.

왜 진단이 먼저인가:
  검색 점수는 쿼리 벡터와 아이템 벡터의 내적이다. 방향 u를 넓히면 점수에 더해지는 몫은
  (쿼리의 u 성분) × (아이템의 u 성분)에 비례한다. 그런데 worklog에는 쿼리의 u 성분이
  −0.037~+0.003이고 록 쿼리조차 −0.004였다고 적혀 있다. 쿼리 쪽이 energy를 담고 있지
  않으면 축을 넓혀도 모든 쿼리를 같은 쪽으로 밀 뿐이다. 그래서 아이템에서 학습한
  프로브를 쿼리에 대 보고, 예측값이 쿼리의 뜻과 같은 순서로 나오는지부터 확인한다.

출력:
  - 프로브의 교차검증 R², 아이템 사영 표준편차
  - 쿼리 40개(영어)의 프로브 예측 energy와, 그 쿼리로 음악만 검색한 상위 10곡의 실제
    평균 energy — 두 열이 같은 순서로 움직이는지(스피어만 상관)
  - 평가 CSV와 같은 검색 결과가 나오는지(인덱스·체크포인트 짝 확인)

쿼리 쌍 시험 (--pairs, 2026-10-09):
  위의 쿼리 40개에는 "원하는 energy"라는 정답이 없고, 예측 energy와 검색 결과 energy의
  상관은 순환에 가깝다(쿼리가 u 방향으로 크면 u 방향으로 큰 곡이 기계적으로 올라온다).
  그래서 energy 표현만 다르고 나머지는 같은 쿼리 쌍(low/high)을 만들어, 정답을 설계로
  고정한다. 두 가지를 본다.
    (1) high 쪽 q·u가 low 쪽보다 큰가 — 쿼리 인코더가 energy의 뜻을 u에 담는가.
    (2) high 쪽 음악 검색 상위10의 실제 energy가 더 높은가 — 그 뜻이 검색을 바꾸는가.
        이쪽은 프로브를 쓰지 않는다.
  각각 양측 부호검정.

사용 (beast):
  python scripts/energy_axis_probe.py --model models/best_stage2_qlora_off.pt \\
      --index indexes_qlora_off --check-csv experiments/eval_lang_ext_qlora_off.csv
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

FEATURE = "energy"


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float):
    """닫힌 형태 릿지. 절편은 따로 두고 X·y를 중심화한 뒤 푼다."""
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
    return w, ym - xm @ w


def cv_r2(X, y, lam, folds=5, seed=0):
    idx = np.random.default_rng(seed).permutation(len(y))
    pred = np.empty_like(y)
    for f in np.array_split(idx, folds):
        tr = np.setdiff1d(idx, f)
        w, b = fit_ridge(X[tr], y[tr], lam)
        pred[f] = X[f] @ w + b
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def spearman(a, b):
    ra = pd.Series(a).rank().values; rb = pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--canonical", default="data/canonical/music_canonical.csv")
    ap.add_argument("--check-csv", help="같은 체크포인트·인덱스로 만든 평가 CSV (재현 확인용)")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default="experiments/energy_axis_probe.csv")
    ap.add_argument("--pairs", help="pair,style,low,high CSV — energy만 다른 쿼리 쌍")
    ap.add_argument("--pairs-out", default="experiments/energy_query_pairs_result.csv")
    args = ap.parse_args()

    # ── 1. 아이템 쪽: 프로브 학습 ────────────────────────────────────────────
    z = torch.load(f"{args.index}/music_embeddings.pt", map_location="cpu", weights_only=False)
    meta = pd.read_parquet(f"{args.index}/music_meta.parquet")
    canon = pd.read_csv(args.canonical, usecols=["id", FEATURE]).drop_duplicates("id")
    lab = meta[["item_id"]].reset_index().merge(canon, left_on="item_id", right_on="id", how="inner")
    lab = lab[lab[FEATURE].notna()]
    X = z[lab["index"].values].numpy().astype(np.float64)
    y = lab[FEATURE].values.astype(np.float64)
    r2 = cv_r2(X, y, args.lam)
    w, b = fit_ridge(X, y, args.lam)
    u = w / np.linalg.norm(w)
    proj = X @ u
    print(f"[아이템] 라벨 있는 곡 {len(y):,} / 인덱스 {len(meta):,}")
    print(f"  프로브 교차검증 R² = {r2:.3f}  (worklog 2-6: 0.598)")
    print(f"  방향 u 사영: 평균 {proj.mean():+.4f}, 표준편차 {proj.std():.4f}  (worklog: 0.007)")
    print(f"  사영과 실제 energy의 상관 = {np.corrcoef(proj, y)[0, 1]:.3f}")

    # ── 2. 쿼리 쪽 ──────────────────────────────────────────────────────────
    os.environ["EVAL_MODEL_PATH"] = args.model
    os.environ["EVAL_INDEX_DIR"] = args.index
    from scripts import eval_lang as E
    model = E.load_model(args.model)
    indexes = {"music": (z, meta.assign(_domain="music"))}
    energy_of = dict(zip(canon["id"].astype(str), canon[FEATURE]))

    rows = []
    for style_code, pair_id, lang, query in E.QUERIES:
        if lang != "en":
            continue
        with torch.no_grad():
            q = F.normalize(model.encode_query([query]), p=2, dim=1).cpu().numpy()[0]
        res = E.search(model, indexes, query, "music", E.TOP_K)
        got = [energy_of.get(r["item_id"]) for r in res]
        got = [g for g in got if g is not None and not np.isnan(g)]
        rows.append({"style": style_code, "pair_id": pair_id, "query": query,
                     "q_proj": float(q @ u), "q_pred": float(q @ w + b),
                     "top10_energy": float(np.mean(got)) if got else np.nan,
                     "top1_item": res[0]["item_id"] if res else ""})
    df = pd.DataFrame(rows).sort_values("q_pred")
    df.to_csv(args.out, index=False)

    print(f"\n[쿼리] 영어 {len(df)}개 — 프로브 예측 energy 순")
    print(f"  q·u 범위 {df.q_proj.min():+.4f} ~ {df.q_proj.max():+.4f} "
          f"(아이템 사영 평균 {proj.mean():+.4f})")
    print(f"  예측 energy 범위 {df.q_pred.min():.3f} ~ {df.q_pred.max():.3f} "
          f"(아이템 실제 평균 {y.mean():.3f})")
    print(f"  예측 energy vs 상위10 실제 energy 스피어만 = "
          f"{spearman(df.q_pred, df.top10_energy):.3f}")
    with pd.option_context("display.width", 160, "display.max_colwidth", 60):
        print(df[["style", "query", "q_proj", "q_pred", "top10_energy"]].to_string(index=False))

    # ── 2-b. 쿼리 쌍 시험 ───────────────────────────────────────────────────
    if args.pairs:
        def probe_query(text):
            with torch.no_grad():
                qv = F.normalize(model.encode_query([text]), p=2, dim=1).cpu().numpy()[0]
            res = E.search(model, indexes, text, "music", E.TOP_K)
            got = [energy_of.get(r["item_id"]) for r in res]
            got = [g for g in got if g is not None and not np.isnan(g)]
            return float(qv @ u), float(np.mean(got)) if got else np.nan

        pr = pd.read_csv(args.pairs)
        out = []
        for r in pr.itertuples(index=False):
            lo_u, lo_e = probe_query(r.low)
            hi_u, hi_e = probe_query(r.high)
            out.append({"pair": r.pair, "style": r.style, "low": r.low, "high": r.high,
                        "low_qu": lo_u, "high_qu": hi_u, "d_qu": hi_u - lo_u,
                        "low_top10_energy": lo_e, "high_top10_energy": hi_e,
                        "d_energy": hi_e - lo_e})
        po = pd.DataFrame(out)
        po.to_csv(args.pairs_out, index=False)

        def sign_p(k, n):
            from math import comb
            tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
            return min(1.0, 2 * tail)

        print(f"\n[쿼리 쌍] {len(po)}쌍 — high가 low보다 큰 쌍의 수")
        for lab, sub in (("전체", po), ("E01~E14 (energy 표현만 바꿈)", po[po["pair"] <= "E14"])):
            n = len(sub)
            k1 = int((sub.d_qu > 0).sum()); k2 = int((sub.d_energy > 0).sum())
            print(f"  {lab}: q·u {k1}/{n} (p={sign_p(k1, n):.4f}, 평균 차 {sub.d_qu.mean():+.4f})"
                  f" | 상위10 energy {k2}/{n} (p={sign_p(k2, n):.4f}, 평균 차 {sub.d_energy.mean():+.3f})")
        print(f"  참고: 아이템 사영 표준편차 {proj.std():.4f}, 쿼리 40개 q·u 표준편차 {df.q_proj.std():.4f}")
        with pd.option_context("display.width", 170, "display.max_colwidth", 42):
            print(po[["pair", "low", "high", "d_qu", "low_top10_energy", "high_top10_energy"]]
                  .round(3).to_string(index=False))

    # ── 3. 재현 확인 ────────────────────────────────────────────────────────
    if args.check_csv:
        c = pd.read_csv(args.check_csv)
        c = c[(c["lang"] == "en") & (c["domain_filter"] == "music") & (c["rank"] == 1)]
        ref = dict(zip(c["query"], c["item_id"].astype(str)))
        same = sum(ref.get(r.query) == str(r.top1_item) for r in df.itertuples())
        print(f"\n[재현] 음악 필터 top-1이 평가 CSV와 같은 쿼리: {same}/{len(df)}")


if __name__ == "__main__":
    main()
