"""사람 라벨용 쌍 시트를 만든다 — 어휘가 오도하는 경우를 표적으로.

왜 무작위 400건이 아닌가:
  쌍대 판정기는 어휘 단서가 **없을 때**(92.6%, 32쌍) 그리고 **양쪽이 같은 정도로 겹칠
  때**(100%, 9쌍) 사람과 맞는다. 확인되지 않은 것은 **어휘가 반대 방향을 가리킬 때**다 —
  6월 라벨에 그런 쌍이 2개뿐이라 1승 1패로 끝났다. 절대 등급 판정은 그 경우 어휘를
  따라간다는 것이 확인돼 있다(어휘 0에서 사람 o율 41.0% vs 판정기 2.7%).
  4-1 하드 네거티브가 정확히 그 축을 겨냥하므로, 4단계 전에 이 칸을 채워야 한다.

쌍을 어떻게 고르는가 — **판정기를 쓰지 않는다(순환 논증 방지).**
  같은 쿼리의 검색 결과에서 (가) 판정기가 보는 본문에 쿼리 어휘가 많이 겹치는 아이템과
  (나) 하나도 겹치지 않는 아이템을 짝짓는다. 선택 기준은 어휘 겹침 수와 검색 순위뿐이다.
  사람이 (가)를 고르면 "어휘는 실제 신호"이고, (나)를 고르면 그것이 찾던 역방향 쌍이다.

시트에는 제목을 가린 본문만 들어간다 — 판정기가 보는 것과 같아야 비교가 성립한다.
좌우는 무작위로 섞고 어느 쪽이 어휘 겹침인지는 **정답지 파일에만** 남긴다.

사용:
  python scripts/build_label_sheet.py --n 60
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
    ap.add_argument("--runs", nargs="+",
                    default=["experiments/eval_lang_ext_qlora_off.csv"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--per-query", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--out-prefix", default="experiments/label_sheet_20260911")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.runs], ignore_index=True)
    df = df[df["lang"] == "en"]
    need = {(str(d), str(i)) for d, i in zip(df.result_domain, df.item_id)}
    lookup = build_item_lookup(True, need)          # 제목 가림 = 판정기가 보는 것

    df["text"] = [lookup.get((str(d), str(i)), ("", True))[0]
                  for d, i in zip(df.result_domain, df.item_id)]
    df["thin"] = [lookup.get((str(d), str(i)), ("", True))[1]
                  for d, i in zip(df.result_domain, df.item_id)]
    df["ov"] = [len(content_words(q) & content_words(t))
                for q, t in zip(df["query"], df["text"])]
    # 정보가 한 줄뿐인 아이템은 뺀다 — 사람도 판정기도 판단할 근거가 없다.
    df = df[(df.text != "") & (~df.thin)]

    rng = random.Random(args.seed)
    pairs = []
    for q, g in df.groupby("query"):
        g = g.drop_duplicates("item_id")
        hi = g[g.ov >= 1].sort_values(["ov", "rank"], ascending=[False, True])
        lo = g[g.ov == 0].sort_values("rank")
        for k in range(min(args.per_query, len(hi), len(lo))):
            a, b = hi.iloc[k], lo.iloc[k]
            flip = rng.random() < 0.5          # 좌우 무작위
            first, second = (b, a) if flip else (a, b)
            pairs.append({
                "query": q, "style": a["style"],
                "후보1": first["text"].replace("\n", " / "),
                "후보2": second["text"].replace("\n", " / "),
                "_ov_side": 2 if flip else 1,   # 어휘가 겹치는 쪽
                "_ov_count": int(a["ov"]), "_ov_rank": int(a["rank"]),
                "_zero_rank": int(b["rank"]),
                "_ov_item": f"{a['result_domain']}:{a['item_id']}",
                "_zero_item": f"{b['result_domain']}:{b['item_id']}",
            })
    rng.shuffle(pairs)
    pairs = pairs[:args.n]
    for i, p in enumerate(pairs, 1):
        p["pair_id"] = f"L{i:03d}"

    sheet = pd.DataFrame(pairs)[["pair_id", "query", "후보1", "후보2"]]
    sheet["선택"] = ""          # 1 / 2 / ? (비슷하거나 모르겠음)
    sheet["메모"] = ""
    key = pd.DataFrame(pairs)[["pair_id", "style", "query", "_ov_side", "_ov_count",
                               "_ov_rank", "_zero_rank", "_ov_item", "_zero_item"]]

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    sheet.to_csv(f"{args.out_prefix}.csv", index=False, encoding="utf-8-sig")
    key.to_csv(f"{args.out_prefix}_key.csv", index=False, encoding="utf-8-sig")

    print(f"쌍 {len(sheet)}개 → {args.out_prefix}.csv (정답지: _key.csv)")
    print(f"  어휘 겹침 수 분포: {dict(key._ov_count.value_counts().sort_index())}")
    print(f"  좌우 배치: 1번 자리 {int((key._ov_side==1).sum())} / 2번 자리 {int((key._ov_side==2).sum())}")
    # key["style"]로 써야 한다 — key.style은 pandas의 Styler 속성이다.
    print(f"  스타일: {dict(key['style'].value_counts())}")


if __name__ == "__main__":
    main()
