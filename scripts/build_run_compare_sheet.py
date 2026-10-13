"""두 검색 실행을 사람이 비교하는 쌍 시트를 만든다.

왜 필요한가 (docs/report_judge_validation_20261009.md):
  로컬 판정기는 "단어가 겹치는 후보가 오답인" 쌍을 잡지 못한다(역방향 정확도 35~55%,
  합격선 75%). 그래서 어휘 지름길을 겨냥한 실험 — 제목·필드명 제거 증강, 4-1 하드
  네거티브 — 은 판정기 대신 사람이 두 실행의 결과를 직접 비교해 평가한다.

쌍을 어떻게 만드는가:
  (쿼리, 도메인필터) 목록마다 두 실행이 **서로 다르게 뽑은** 아이템을 순위 순으로 짝짓는다
  (judge_pairwise.build_item_units와 같은 방식). 공통 아이템은 비교할 필요가 없다.
  - 목록마다 최대 1쌍 — 특정 쿼리가 결과를 좌우하지 않게.
  - 몇 번째 짝을 쓸지도 무작위 — 상위 순위만 보지 않게.
  - 좌우 무작위. 어느 쪽이 어느 실행인지는 정답지(_key.csv)에만.
  - 정보가 한 줄뿐인 아이템이 끼면 뺀다 — 사람이 판단할 근거가 없다.
  정답지에는 각 후보가 쿼리와 겹치는 내용어 수도 남긴다. 전체 선호율과 함께
  "A는 겹치고 B는 안 겹치는" 쌍만 따로 볼 수 있게 하기 위해서다.

사용:
  python scripts/build_run_compare_sheet.py \\
      --a base:experiments/eval_lang_ext_qlora_off.csv \\
      --b titledrop:experiments/eval_lang_ext_dropout03.csv \\
      --n 120 --out-prefix experiments/compare_titledrop_20261009
"""
import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.analyze_title_bias import content_words
from scripts.judge_eval_relevance import build_item_lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="이름:CSV — 기준 실행")
    ap.add_argument("--b", required=True, help="이름:CSV — 비교 실행")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    name_a, path_a = args.a.split(":", 1)
    name_b, path_b = args.b.split(":", 1)
    fa = pd.read_csv(path_a); fb = pd.read_csv(path_b)
    fa = fa[fa["lang"] == args.lang]; fb = fb[fb["lang"] == args.lang]
    need = {(str(d), str(i)) for f in (fa, fb) for d, i in zip(f.result_domain, f.item_id)}
    lookup = build_item_lookup(True, need)          # 제목 가림 = 라벨 시트와 같은 조건

    def info(r):
        text, thin = lookup.get((str(r.result_domain), str(r.item_id)), ("", True))
        return text, thin

    rng = random.Random(args.seed)
    # 실행마다 query_id 접미사가 다를 수 있어 (스타일, 쿼리번호, 도메인필터)로 맞춘다
    keys = ["style", "pair_id", "domain_filter"]
    gb_all = {k: g for k, g in fb.groupby(keys)}
    cands = []
    for k, ga in fa.groupby(keys):
        gb = gb_all.get(k)
        if gb is None:
            continue
        ga = ga.sort_values("rank").head(args.topk)
        gb = gb.sort_values("rank").head(args.topk)
        sa = set(zip(ga.result_domain.astype(str), ga.item_id.astype(str)))
        sb = set(zip(gb.result_domain.astype(str), gb.item_id.astype(str)))
        onlya = [r for r in ga.itertuples() if (str(r.result_domain), str(r.item_id)) not in sb]
        onlyb = [r for r in gb.itertuples() if (str(r.result_domain), str(r.item_id)) not in sa]
        usable = []
        for ra, rb in zip(onlya, onlyb):
            (ta, thin_a), (tb, thin_b) = info(ra), info(rb)
            if ta and tb and not thin_a and not thin_b:
                usable.append((ra, rb, ta, tb))
        if usable:
            cands.append(rng.choice(usable))        # 목록당 1쌍, 짝 위치도 무작위

    rng.shuffle(cands)
    cands = cands[:args.n]

    sheet, key = [], []
    for i, (ra, rb, ta, tb) in enumerate(cands, 1):
        b_first = rng.random() < 0.5
        c1, c2 = (tb, ta) if b_first else (ta, tb)
        pid = f"R{i:03d}"
        qw = content_words(ra.query)
        sheet.append({"pair_id": pid, "query": ra.query,
                      "후보1": c1.replace("\n", " / "), "후보2": c2.replace("\n", " / "),
                      "선택": "", "메모": ""})
        key.append({"pair_id": pid, "style": ra.style, "query": ra.query,
                    "domain_filter": ra.domain_filter, "b_side": 1 if b_first else 2,
                    "a_item": f"{ra.result_domain}:{ra.item_id}", "a_rank": int(ra.rank),
                    "a_ov": len(qw & content_words(ta)),
                    "b_item": f"{rb.result_domain}:{rb.item_id}", "b_rank": int(rb.rank),
                    "b_ov": len(qw & content_words(tb)),
                    "a_name": name_a, "b_name": name_b})

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    pd.DataFrame(sheet).to_csv(f"{args.out_prefix}.csv", index=False, encoding="utf-8-sig")
    k = pd.DataFrame(key)
    k.to_csv(f"{args.out_prefix}_key.csv", index=False, encoding="utf-8-sig")

    print(f"쌍 {len(k)}개 → {args.out_prefix}.csv (정답지: _key.csv)")
    print(f"  {name_b} 쪽 위치: 1번 {int((k.b_side == 1).sum())} / 2번 {int((k.b_side == 2).sum())}")
    print(f"  도메인필터: {k.domain_filter.value_counts().to_dict()}")
    print(f"  스타일: {k['style'].value_counts().to_dict()}")
    print(f"  {name_a}만 단어 겹침: {int(((k.a_ov > 0) & (k.b_ov == 0)).sum())} / "
          f"{name_b}만 겹침: {int(((k.b_ov > 0) & (k.a_ov == 0)).sum())} / "
          f"둘 다: {int(((k.a_ov > 0) & (k.b_ov > 0)).sum())} / "
          f"둘 다 없음: {int(((k.a_ov == 0) & (k.b_ov == 0)).sum())}")


if __name__ == "__main__":
    main()
