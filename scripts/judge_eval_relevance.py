"""
검색 결과의 관련성을 LLM으로 o/a/x 판정한다 (6월 baseline과 신규 실행을 같은 잣대로).

왜 필요한가:
  eval_lang.py가 내놓는 score는 모델이 자기 임베딩으로 계산한 코사인 유사도다. 학습이
  바뀌면 임베딩 공간 자체가 달라지므로 절대값을 실행 간에 비교할 수 없다. 6월 리포트의
  "poet 0.86"은 전혀 다른 축 — 사람과 Claude가 매긴 관련성 등급(o=2/a=1/x=0)의 평균이다.
  따라서 신규 실행을 6월과 비교하려면 **같은 판정기로 양쪽을 다시 매겨야** 한다.

설계 — 비교를 공정하게 만드는 세 가지:

  (1) 블라인드. 두 실행의 행을 섞고, 프롬프트에 어느 실행인지 넣지 않는다.

  (2) 판정 근거는 **원문 필드**(제목·장르·감독/저자·줄거리)만 쓴다. description_synth를
      쓰면 안 된다 — 신규 실행의 아이템은 mood 어휘가 풍부한 합성 설명을 갖고 있고
      6월 아이템은 사실 위주 원문뿐이라, 판정기가 신규 쪽을 체계적으로 유리하게 볼 수
      있다. 우리가 평가하려는 것은 "합성 설명이 잘 써졌나"가 아니라 "이 아이템이 이
      쿼리에 맞나"이므로, 아이템의 정체는 원문으로 알려주는 것이 맞다.

  (3) 6월 book은 ID 체계가 다르다(구 book_canonical.csv의 ASIN, v2는 kdl_/gr_/bx_).
      구 CSV를 함께 읽어 해결한다 — 실측 해결률 98.6%.

  판정기 신뢰도는 6월 행에 남아 있는 사람 라벨과의 일치율로 함께 보고한다. 이 값이
  낮으면 판정 결과 자체를 믿을 수 없다는 뜻이므로 리포트 맨 앞에 출력한다.

사용:
  python scripts/judge_eval_relevance.py \
      --runs 2026-06:experiments/eval_lang_20260618_en_valids.csv \
             2026-08:experiments/eval_lang_20260810.csv
"""

import argparse
import json
import os
import re

import pandas as pd

CACHE_PATH = "data/cache/judge_relevance_cache.json"
GRADE_VALUE = {"o": 2.0, "a": 1.0, "x": 0.0}

PROMPT = (
    "[Task]\n"
    "You are evaluating a search result. A user searched with an evocative, vibe-based "
    "query. Judge whether the retrieved {noun} is a good match for that query.\n\n"
    "[Query]\n{query}\n\n"
    "[Retrieved {noun_title}]\n{item}\n\n"
    # 등급 정의는 사람 라벨의 분포에 맞춰 보정했다. 첫 판정에서 'a'가 0.6%로 붕괴해
    # (사람은 26.7%) 사실상 o/x 이진 판정이 됐고, 사람이 '애매'로 본 85건 중 51건을
    # x로 떨어뜨려 전반적으로 가혹해졌다. 'a'를 기본값으로 두고 o/x가 되려면 근거를
    # 요구하는 방향으로 바꾼다.
    "[Grades]\n"
    "a = the DEFAULT grade. Use it whenever the work is plausibly connected to the query — "
    "a shared theme, a compatible mood, an adjacent subject — but you would not call it a "
    "showcase result. Roughly a quarter of results should land here.\n"
    "o = reserve for a strong match. The work's central mood or subject is what the query "
    "asks for; a user would nod at seeing it first.\n"
    "x = reserve for a clear miss. Nothing in the work connects to the query's mood or "
    "subject — not the theme, not the tone, not the setting.\n\n"
    "[Rules]\n"
    "- Judge the WORK itself, not how well it is described.\n"
    "- Vibe queries are metaphorical. A work matches if its atmosphere or theme resonates, "
    "even if no words overlap literally.\n"
    "- Do not reward mere keyword overlap with the query.\n"
    "- If the source material below is thin, judge from what is given; do not assume. Thin "
    "source alone is not grounds for x — grade what is there.\n"
    "- When torn between two grades, choose a.\n\n"
    "[Output]\nExactly one character: o, a, or x. No explanation."
)


# ── 아이템 원문 조회 ──────────────────────────────────────────────────────────

def _clean(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    return "" if s in ("nan", "None", "[]", "no") else s


def _json_list(v, limit=3) -> str:
    try:
        items = json.loads(str(v))
        if isinstance(items, list) and items:
            return ", ".join(str(i) for i in items[:limit])
    except Exception:
        pass
    return _clean(v)


def build_item_lookup() -> dict:
    """(domain, item_id) → 원문 기반 설명. 합성 설명은 의도적으로 쓰지 않는다(설계 (2))."""
    lookup: dict[tuple[str, str], str] = {}

    m = pd.read_csv("data/canonical/movie_canonical.csv", low_memory=False)
    for r in m.itertuples():
        parts = [f"Title: {_clean(r.Title)}", f"Genre: {_clean(r.Genre)}"]
        if d := _json_list(getattr(r, "director", None)):
            parts.append(f"Director: {d}")
        if a := _json_list(getattr(r, "actor", None), 4):
            parts.append(f"Cast: {a}")
        if ov := _clean(getattr(r, "text", None)):
            parts.append(f"Overview: {ov[:600]}")
        lookup[("movie", str(r.imdbId))] = "\n".join(parts)

    mu = pd.read_csv("data/canonical/music_canonical.csv", low_memory=False)
    for r in mu.itertuples():
        parts = [f"Track: {_clean(r.name)}", f"Artist: {_json_list(getattr(r, 'artists', None))}",
                 f"Album: {_clean(getattr(r, 'album_name', None))}",
                 f"Genre: {_clean(getattr(r, 'genre', None))}"]
        desc = _clean(getattr(r, "description", None))
        lyr = _clean(getattr(r, "lyrics", None))
        if desc:
            parts.append(f"Description: {desc[:500]}")
        elif lyr:
            parts.append(f"Lyrics excerpt: {lyr[:400]}")
        lookup[("music", str(r.id))] = "\n".join(parts)

    # book은 두 세대를 모두 읽는다 — 6월 결과는 구 ASIN, 신규는 v2의 kdl_/gr_/bx_.
    for path in ("data/canonical/book_canonical_v2.csv", "data/canonical/book_canonical.csv"):
        if not os.path.exists(path):
            print(f"  [warn] {path} 없음 — 해당 세대 book 항목은 판정에서 빠진다")
            continue
        b = pd.read_csv(path, low_memory=False)
        for r in b.itertuples():
            key = ("book", str(r.asin))
            if key in lookup:
                continue
            parts = [f"Title: {_clean(r.title)}", f"Author: {_clean(getattr(r, 'author', None))}",
                     f"Category: {_clean(getattr(r, 'category_name', None))}"]
            blurb = _clean(getattr(r, "description_clean", None)) or _clean(getattr(r, "description", None))
            if blurb:
                parts.append(f"Blurb: {blurb[:600]}")
            lookup[key] = "\n".join(parts)

    return lookup


# ── 판정 ─────────────────────────────────────────────────────────────────────

NOUNS = {"movie": ("film", "Film"), "music": ("track", "Track"), "book": ("book", "Book")}


def parse_grade(raw: str) -> str | None:
    m = re.search(r"\b([oax])\b", str(raw).strip().lower())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="라벨:CSV경로 형식")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default="experiments/judge_report.txt")
    args = ap.parse_args()

    frames = []
    for spec in args.runs:
        label, path = spec.split(":", 1)
        df = pd.read_csv(path)
        df["run_label"] = label
        df = df[df["lang"] == "en"] if "lang" in df.columns else df
        frames.append(df)
        print(f"[{label}] {len(df):,}행 ({path})")
    data = pd.concat(frames, ignore_index=True)

    print("아이템 원문 조회 준비...")
    lookup = build_item_lookup()

    data["item_text"] = [
        lookup.get((str(d), str(i)), "") for d, i in zip(data["result_domain"], data["item_id"])
    ]
    missing = int((data["item_text"] == "").sum())
    if missing:
        print(f"  [warn] 원문을 못 찾은 행 {missing:,}건 — 판정에서 제외")
    data = data[data["item_text"] != ""].reset_index(drop=True)

    cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
    data["judge_key"] = [
        f"{d}|{i}|{q}" for d, i, q in zip(data["result_domain"], data["item_id"], data["query"])
    ]
    todo = data[~data["judge_key"].isin(cache)].drop_duplicates("judge_key")
    print(f"판정 대상 {len(todo):,}건 (캐시 {len(cache):,}건 재사용)")

    if len(todo):
        from scripts.vllm_runner import VLLMRunner, chunks
        # 블라인드: 실행 라벨을 프롬프트에 넣지 않고, 순서도 섞어 배치 구성이 실행별로
        # 쏠리지 않게 한다(설계 (1)).
        todo = todo.sample(frac=1.0, random_state=42).reset_index(drop=True)
        runner = VLLMRunner(args.model_id, max_new_tokens=4)

        done = 0
        for batch in chunks(list(todo.itertuples()), args.batch):
            items = []
            for r in batch:
                noun, noun_title = NOUNS[r.result_domain]
                items.append((PROMPT.format(noun=noun, noun_title=noun_title,
                                            query=r.query, item=r.item_text), None))
            for r, raw in zip(batch, runner.generate(items)):
                g = parse_grade(raw)
                if g:
                    cache[r.judge_key] = g
            done += len(batch)
            json.dump(cache, open(CACHE_PATH, "w"))
            print(f"  진행 {done:,}/{len(todo):,}", flush=True)

    data["grade"] = data["judge_key"].map(cache)
    data["value"] = data["grade"].map(GRADE_VALUE)
    judged = data[data["value"].notna()]

    lines = [f"LLM 관련성 판정 리포트 ({args.model_id})", "=" * 68, ""]

    # 판정기 신뢰도 — 6월 행의 사람/Claude 라벨과 얼마나 맞는가. 낮으면 이하 수치를
    # 믿을 수 없으므로 맨 앞에 둔다.
    if "Validity" in judged.columns:
        v = judged[judged["Validity"].notna()]
        if len(v):
            agree = (v["Validity"].str.strip() == v["grade"]).mean()
            adj = (v["Validity"].map(GRADE_VALUE) - v["value"]).abs().le(1).mean()
            lines += ["## 0. 판정기 신뢰도 (6월 사람/Claude 라벨 대비)",
                      f"  완전일치 {agree:.1%} / 인접일치(±1등급) {adj:.1%} / 대상 {len(v):,}행",
                      "  ※ 완전일치가 낮으면 이하 절대값보다 실행 간 '차이'만 보아야 한다.", ""]

    lines += ["## 1. 실행별 종합", f"  {'실행':14s} {'행':>6s} {'avg':>7s} {'±SE':>6s} "
              f"{'o율':>7s} {'a율':>7s} {'x율':>7s}"]
    for run, g in judged.groupby("run_label"):
        se = g["value"].std(ddof=1) / (len(g) ** 0.5)
        lines.append(f"  {run:14s} {len(g):>6,} {g['value'].mean():>7.3f} {se:>6.3f} "
                     f"{(g['grade']=='o').mean():>7.1%} {(g['grade']=='a').mean():>7.1%} "
                     f"{(g['grade']=='x').mean():>7.1%}")

    # 확장 쿼리(pair_id 5~10)는 6월에 없다. 섞어서 평균 내면 baseline 대비 수치가 무의미해지므로
    # 같은 쿼리 집합끼리만 비교할 수 있도록 분리해 둔다.
    if "pair_id" in judged.columns and (judged["pair_id"] > 4).any():
        base = judged[judged["pair_id"] <= 4]
        lines += ["", "  [6월과 공통인 쿼리(pair_id 1~4)로 한정]"]
        for run, g in base.groupby("run_label"):
            se = g["value"].std(ddof=1) / (len(g) ** 0.5)
            lines.append(f"  {run:14s} {len(g):>6,} {g['value'].mean():>7.3f} {se:>6.3f}")

    lines += ["", "## 2. 스타일별 avg_score (핵심 — poet이 6월 최하였다)"]
    piv = judged.pivot_table(index="style", columns="run_label", values="value", aggfunc="mean")
    runs = list(piv.columns)
    # 마지막 두 실행의 차이를 표준오차로 나눠 노이즈와 구분한다. 스타일당 표본이 적으면
    # 0.1 수준의 차이는 판정할 수 없다 — 8/13 실행에서 poet +0.138이 1.5 SE였다.
    lines.append("  " + f"{'스타일':16s}" + "".join(f"{r:>14s}" for r in runs) +
                 f"{'변화':>9s}{'SE배수':>8s}")
    for style, row in piv.iterrows():
        cells = "".join(f"{row[r]:>14.3f}" for r in runs)
        tail = ""
        if len(runs) > 1:
            a = judged[(judged.style == style) & (judged.run_label == runs[-2])]["value"]
            b = judged[(judged.style == style) & (judged.run_label == runs[-1])]["value"]
            delta = b.mean() - a.mean()
            se = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 0.5
            tail = f"{delta:>+9.3f}{(abs(delta) / se if se else 0):>8.1f}"
        lines.append("  " + f"{style:16s}" + cells + tail)
    lines.append("  ※ SE배수 2 이상이면 유의, 1 미만이면 노이즈로 본다 (마지막 두 실행 기준)")

    lines += ["", "## 3. 도메인별 avg_score"]
    piv2 = judged.pivot_table(index="domain_filter", columns="run_label", values="value", aggfunc="mean")
    lines.append("  " + f"{'도메인':16s}" + "".join(f"{r:>10s}" for r in runs) + f"{'변화':>9s}")
    for dom, row in piv2.iterrows():
        delta = row[runs[-1]] - row[runs[0]] if len(runs) > 1 else float("nan")
        lines.append("  " + f"{dom:16s}" + "".join(f"{row[r]:>10.3f}" for r in runs) +
                     (f"{delta:>+9.3f}" if len(runs) > 1 else ""))

    lines += ["", "## 4. 쿼리별 (신규 실행 기준 오름차순 = 개선 우선순위)"]
    piv3 = judged.pivot_table(index=["style", "query"], columns="run_label", values="value", aggfunc="mean")
    for (style, query), row in piv3.sort_values(runs[-1]).iterrows():
        vals = "".join(f"{row[r]:>8.2f}" if pd.notna(row[r]) else f"{'-':>8s}" for r in runs)
        lines.append(f"  {style:14s}{vals}  {query[:48]}")

    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    judged.drop(columns=["item_text"]).to_csv(args.out.replace(".txt", "_rows.csv"), index=False)
    print(f"\n리포트 → {args.out}")


if __name__ == "__main__":
    main()
