#!/usr/bin/env python
"""제목 매칭 편향과 도메인 쏠림을 실행(run) 간에 비교한다.

왜 필요한가 — 판정 평균(judge_report)만으로는 세션 19의 21절 [1][2]가 고쳐졌는지 알 수
없다. 두 문제는 "점수가 낮다"가 아니라 "점수가 엉뚱한 이유로 높다"이기 때문이다.

  [1] 제목 매칭 편향 : 쿼리 단어를 제목에 그대로 담은 아이템이 상위에 오고, 판정기도
      그것을 높게 준다. 뜻이 반대여도 그렇다("Warmth left in a room after the lights
      go out" → "Stay Or Leave"). 겹침 비율과 **겹칠 때의 판정 평균 차이**를 함께 봐야
      편향인지 우연인지 갈린다.
  [2] 도메인 쏠림 : 전역 비중은 인덱스 비중과 비슷한데 쿼리마다 한두 도메인이 상위를
      독식한다. 그래서 전체 분포가 아니라 **쿼리별 집중도**를 센다.

기준은 통합검색(domain_filter=all)·영어·top-10 = 쿼리 40개 x 10 = 400건이다. 세션 19가
이 기준으로 쟀으므로 비교 가능하도록 맞췄다. base(frozen_noLoRA) 실행을 넣으면 그때 수치
(65건 16.2%, 겹침 1.062 / 안겹침 0.752)가 그대로 재현된다 — 스크립트 검증에 쓸 것.

사용:
  python scripts/analyze_title_bias.py \
      experiments/eval_lang_ext_qlora_off.csv:base \
      experiments/eval_lang_ext_dropout03.csv:dropout03
  # 판정 점수까지 보려면 --judge 로 *_rows.csv 를 함께 준다
"""
import argparse
import re
from collections import Counter

import pandas as pd

# 불용어를 빼는 이유: the/of 같은 단어는 어느 제목에나 있어 전부 "겹침"으로 잡힌다.
# 3자 미만도 같은 이유로 뺀다.
STOP = set("""a an the of in on at to for with and or but is are was were be been being
this that these those it its as by from into over under after before between through
about like without within during your my his her their our you i we they he she them
what how much made me do does did not no nor so than then there here when where
which who whom whose will would can could should may might must have has had having
one two some any all more most other another such own same too very just only up down
out off again further once""".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", str(text).lower()) if w not in STOP and len(w) >= 3}


def load(path: str, judge: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lstrip("﻿") for c in df.columns]   # 파일 선두의 BOM
    df = df[(df["lang"] == "en") & (df["domain_filter"] == "all") & (df["rank"] <= 10)].copy()
    df["overlap"] = [bool(content_words(q) & content_words(t))
                     for q, t in zip(df["query"], df["title"])]
    return df


def report(df: pd.DataFrame, label: str) -> None:
    per_query, dom_counts = [], []
    for _, g in df.groupby(["query_id", "query"]):
        per_query.append(int(g["overlap"].sum()))
        dom_counts.append(Counter(g["result_domain"]))

    hits, total = sum(per_query), len(df)
    share = Counter()
    for c in dom_counts:
        share.update(c)

    print(f"\n[{label}]  쿼리 {len(per_query)}개 · 결과 {total}건")
    print(f"  제목 매칭      {hits}건 ({hits / total:.1%})  쿼리별 {dict(sorted(Counter(per_query).items()))}")
    print(f"  한 도메인 8개+ {sum(1 for c in dom_counts if max(c.values()) >= 8)}개 쿼리 · "
          f"10개 전부 {sum(1 for c in dom_counts if max(c.values()) == 10)}개 · "
          f"세 도메인 모두 {sum(1 for c in dom_counts if len(c) == 3)}개")
    print(f"  상위10 평균 도메인 수 {sum(len(c) for c in dom_counts) / len(dom_counts):.2f}  "
          + " ".join(f"{k} {v / total:.1%}" for k, v in sorted(share.items())))

    if "value" in df.columns:
        g = df.groupby("overlap")["value"]
        on, off = g.mean().get(True, float("nan")), g.mean().get(False, float("nan"))
        print(f"  판정 평균      겹침 {on:.3f} (n={int(df['overlap'].sum())}) / "
              f"안겹침 {off:.3f} (n={int((~df['overlap']).sum())})  차이 {on - off:+.3f}")
        print("                 ↑ 이 차이가 클수록 제목 매칭이 보상받고 있다는 뜻이다")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="eval_lang_ext CSV 경로[:라벨]")
    ap.add_argument("--judge", nargs="*", default=[],
                    help="judge_report_*_rows.csv 경로[:run_label] — 판정 점수까지 본다")
    args = ap.parse_args()

    for spec in args.runs:
        path, _, label = spec.partition(":")
        report(load(path), label or path)

    for spec in args.judge:
        path, _, run_label = spec.partition(":")
        df = load(path)
        if run_label:
            df = df[df["run_label"] == run_label]
        report(df, f"{run_label or path} (판정 포함)")


if __name__ == "__main__":
    main()
