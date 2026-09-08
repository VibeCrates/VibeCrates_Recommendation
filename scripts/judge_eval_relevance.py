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

2026-09-08 추가 — 제목 가림(기본값)과 그에 딸린 두 가지:

  (4) **제목 가림.** 이 판정기는 제목이 쿼리와 겹치면 평균 1.062, 안 겹치면 0.752를
      준다. 어휘 지름길을 줄이는 개선을 재려는데 저울이 그 지름길을 보상하고 있으므로,
      제목을 빼고 나머지 필드만 보여준다. `Title:` 줄을 지우는 것만으로는 부족하다 —
      본문이 제 제목을 그대로 부르는 비율이 movie 13.8% / book 19.7% / music 22.6%다
      (2026-09-08 실측, 판정 대상 6,317 아이템 기준). 본문 안의 제목까지 지운다.

  (5) **캐시 서명.** 판정 캐시 키가 (도메인, 아이템, 쿼리)뿐이라 프롬프트를 바꿔도 같은
      키가 나온다. 그대로 두면 가림 조건에서 옛 등급이 todo에서 빠지고 새 조건의 결과인
      척 리포트에 실린다 — 경고 한 줄 없이. 그래서 프롬프트와 렌더 함수의 해시를 키와
      캐시 파일명에 함께 넣어, 조건이 바뀌면 캐시가 저절로 갈린다.

  (6) **설명 결손 행을 빼지 않고 따로 센다.** music의 9.1%(510행)는 description도
      lyrics도 없어 제목을 가리면 Artist·Album·Genre만 남는다. 제외하면 실행마다 빠지는
      행 수가 달라(meanpool 165행 vs centered 58행) 서로 다른 표본의 평균을 비교하게
      된다. 그래서 판정에는 넣고, 리포트에서 주 지표(결손 제외)와 결손 노출률을 나눠
      보고한다. 노출률 자체가 실행의 성질이다.

사용:
  python scripts/judge_eval_relevance.py \
      --runs 2026-06:experiments/eval_lang_20260618_en_valids.csv \
             2026-08:experiments/eval_lang_20260810.csv
  # 제목을 보여주던 옛 조건으로 돌리려면 --no-mask-title
  # 한국어 판정은 --lang ko (검색 결과는 CSV에 이미 들어 있다)
"""

import argparse
import hashlib
import inspect
import json
import os
import re

import pandas as pd

CACHE_DIR = "data/cache"
GRADE_VALUE = {"o": 2.0, "a": 1.0, "x": 0.0}

NOUNS = {"movie": ("film", "Film"), "music": ("track", "Track"), "book": ("book", "Book")}
TITLE_LABEL = {"movie": "Title", "music": "Track", "book": "Title"}

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

# 가림 사실을 알려 주지 않으면 판정기가 "무슨 작품인지 알 수 없다"를 x의 근거로 삼아
# 모든 실행이 함께 내려간다. 그러면 실행 간 차이를 보려던 목적이 그대로 무너진다.
MASK_RULE = (
    "- The title of the work is withheld on purpose. Judge from the remaining fields; do "
    "not penalise the missing title and do not try to guess what the work is called.\n"
)


def build_prompt(mask_title: bool) -> str:
    return PROMPT.replace("[Rules]\n", "[Rules]\n" + MASK_RULE, 1) if mask_title else PROMPT


# ── 제목 가림 ────────────────────────────────────────────────────────────────

_WORD = re.compile(r"[a-z0-9]+")
_YEAR_SUFFIX = re.compile(r"\s*\(\d{4}\)\s*$")


def _title_pattern(title: str):
    """본문에서 제목을 찾기 위한 정규식. 대소문자·구두점·공백 차이를 흡수한다.

    글자 수 4 미만인 제목("I Am", "Up")은 패턴을 만들지 않는다. 그런 제목은 본문 어디에나
    있는 흔한 단어라, 지우면 제목을 가리는 게 아니라 문장을 부순다.
    """
    toks = _WORD.findall(title.lower())
    if not toks or sum(len(t) for t in toks) < 4:
        return None
    body = r"[^a-z0-9]{0,3}".join(re.escape(t) for t in toks)
    # 뒤쪽 경계에서 대문자는 허용한다 — 원문에 공백이 빠져 "Fletch"와 다음 문장이 붙는
    # 경우("FletchHe was...")가 실제로 있다. 소문자를 허용하지 않는 것은 굴절형과
    # 파생어를 지키기 위해서다("Fletcher"는 다른 사람이고 "Volcanoes"는 제목이 아니다).
    # IGNORECASE를 패턴 전체에 걸면 뒤쪽 경계의 [a-z0-9]가 대문자까지 막아 위 의도가
    # 무효가 된다. 대소문자 무시는 본문 부분에만 (?i:…)로 한정한다.
    return re.compile(rf"(?<![A-Za-z0-9])(?i:{body})(?![a-z0-9])")


def scrub_title(text: str, title: str, noun: str) -> str:
    """본문에 남은 제목을 도메인 명사로 바꾼다. 자리표시자가 아니라 실제 명사를 쓰는 것은
    "…is a song about"이 "this track is a song about"으로 자연스럽게 읽히게 하기 위해서다.
    """
    pat = _title_pattern(title)
    return pat.sub(f"this {noun}", text) if pat else text


def render_item(domain: str, title: str, fields: list, mask_title: bool) -> str:
    """(라벨, 값) 목록을 판정 프롬프트에 넣을 한 덩어리로 만든다. 빈 값은 버린다.

    가림 모드에서는 제목 줄을 빼고, **남은 모든 필드**에서 제목을 지운다. 본문만 훑으면
    부족하다 — music은 album_name이 곡 제목과 같은 경우가 8.5%라 앨범 칸으로 샌다.

    이 함수의 소스가 캐시 서명에 들어간다. 렌더 방식을 바꾸면 옛 판정이 자동 폐기된다.
    """
    out = []
    for label, value in fields:
        if not value:
            continue
        if label == TITLE_LABEL[domain]:
            if mask_title:
                continue
        elif mask_title:
            value = scrub_title(value, title, NOUNS[domain][0])
        out.append(f"{label}: {value}")
    return "\n".join(out)


def prompt_signature(mask_title: bool) -> str:
    """프롬프트와 렌더 방식의 해시. 캐시 키와 파일명에 함께 들어간다.

    버전 숫자를 손으로 올리는 방식을 쓰지 않는 이유는, 올리는 것을 잊는 순간 실패가
    조용하기 때문이다 — 옛 등급이 새 조건의 결과로 리포트에 실린다.
    """
    src = "".join([
        build_prompt(mask_title),
        inspect.getsource(render_item),
        inspect.getsource(scrub_title),
        inspect.getsource(_title_pattern),
    ])
    return hashlib.sha1(src.encode()).hexdigest()[:8]


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


def build_item_lookup(mask_title: bool, needed: set | None = None) -> dict:
    """(domain, item_id) → (판정용 원문, 자유 서술이 비었는가).

    합성 설명은 의도적으로 쓰지 않는다(설계 (2)). 두 번째 값은 리포트에서 결손 노출률을
    세는 데 쓴다 — 판정에서 빼지는 않는다(설계 (6)).

    needed를 주면 그 (도메인, id)만 렌더한다. 가림 모드는 아이템마다 정규식을 돌리므로
    canonical 322,893건을 전부 렌더하면 몇 분이 걸린다 — 실제로 판정할 것은 수천 건이다.
    """
    lookup: dict = {}

    def wanted(domain: str, item_id) -> bool:
        return needed is None or (domain, str(item_id)) in needed

    m = pd.read_csv("data/canonical/movie_canonical.csv", low_memory=False)
    for r in m.itertuples():
        if not wanted("movie", r.imdbId):
            continue
        title = _clean(r.Title)
        overview = _clean(getattr(r, "text", None))
        fields = [
            ("Title", title),
            ("Genre", _clean(r.Genre)),
            ("Director", _json_list(getattr(r, "director", None))),
            ("Cast", _json_list(getattr(r, "actor", None), 4)),
            ("Overview", overview[:600]),
        ]
        # 제목의 "(2010)"은 본문에 함께 나오지 않는다. 붙여 두면 본문 매칭이 전부 빗나간다.
        lookup[("movie", str(r.imdbId))] = (
            render_item("movie", _YEAR_SUFFIX.sub("", title), fields, mask_title),
            not overview,
        )

    mu = pd.read_csv("data/canonical/music_canonical.csv", low_memory=False)
    for r in mu.itertuples():
        if not wanted("music", r.id):
            continue
        title = _clean(r.name)
        desc = _clean(getattr(r, "description", None))
        lyr = _clean(getattr(r, "lyrics", None))
        body = ("Description", desc[:500]) if desc else ("Lyrics excerpt", lyr[:400])
        fields = [
            ("Track", title),
            ("Artist", _json_list(getattr(r, "artists", None))),
            ("Album", _clean(getattr(r, "album_name", None))),
            ("Genre", _clean(getattr(r, "genre", None))),
            body,
        ]
        lookup[("music", str(r.id))] = (
            render_item("music", title, fields, mask_title),
            not (desc or lyr),
        )

    # book은 두 세대를 모두 읽는다 — 6월 결과는 구 ASIN, 신규는 v2의 kdl_/gr_/bx_.
    for path in ("data/canonical/book_canonical_v2.csv", "data/canonical/book_canonical.csv"):
        if not os.path.exists(path):
            print(f"  [warn] {path} 없음 — 해당 세대 book 항목은 판정에서 빠진다")
            continue
        b = pd.read_csv(path, low_memory=False)
        for r in b.itertuples():
            key = ("book", str(r.asin))
            if key in lookup or not wanted("book", r.asin):
                continue
            title = _clean(r.title)
            blurb = _clean(getattr(r, "description_clean", None)) or _clean(getattr(r, "description", None))
            fields = [
                ("Title", title),
                ("Author", _clean(getattr(r, "author", None))),
                ("Category", _clean(getattr(r, "category_name", None))),
                ("Blurb", blurb[:600]),
            ]
            lookup[key] = (render_item("book", title, fields, mask_title), not blurb)

    return lookup


# ── 판정 ─────────────────────────────────────────────────────────────────────

def parse_grade(raw: str) -> str | None:
    m = re.search(r"\b([oax])\b", str(raw).strip().lower())
    return m.group(1) if m else None


def _run_table(frame, runs, header):
    """실행별 행 수·평균·SE·등급 분포 한 벌."""
    lines = [header, f"  {'실행':14s} {'행':>6s} {'avg':>7s} {'±SE':>6s} "
             f"{'o율':>7s} {'a율':>7s} {'x율':>7s}"]
    for run in runs:
        g = frame[frame["run_label"] == run]
        if not len(g):
            continue
        se = g["value"].std(ddof=1) / (len(g) ** 0.5)
        lines.append(f"  {run:14s} {len(g):>6,} {g['value'].mean():>7.3f} {se:>6.3f} "
                     f"{(g['grade']=='o').mean():>7.1%} {(g['grade']=='a').mean():>7.1%} "
                     f"{(g['grade']=='x').mean():>7.1%}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="라벨:CSV경로 형식")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default="experiments/judge_report.txt")
    ap.add_argument("--lang", default="en", choices=["en", "ko", "all"],
                    help="판정할 쿼리 언어. ko 결과도 CSV에 이미 들어 있다")
    ap.add_argument("--no-mask-title", dest="mask_title", action="store_false",
                    help="제목을 보여주던 2026-09-08 이전 조건으로 되돌린다")
    ap.set_defaults(mask_title=True)
    args = ap.parse_args()

    sig = prompt_signature(args.mask_title)
    cache_path = os.path.join(CACHE_DIR, f"judge_cache_abs_{sig}.json")
    print(f"판정 조건: 제목 {'가림' if args.mask_title else '노출'} / 언어 {args.lang} / 서명 {sig}")
    print(f"캐시: {cache_path}")

    frames = []
    for spec in args.runs:
        label, path = spec.split(":", 1)
        df = pd.read_csv(path)
        df["run_label"] = label
        if args.lang != "all" and "lang" in df.columns:
            df = df[df["lang"] == args.lang]
        frames.append(df)
        print(f"[{label}] {len(df):,}행 ({path})")
    data = pd.concat(frames, ignore_index=True)
    run_order = [s.split(":", 1)[0] for s in args.runs]

    print("아이템 원문 조회 준비...")
    needed = {(str(d), str(i)) for d, i in zip(data["result_domain"], data["item_id"])}
    lookup = build_item_lookup(args.mask_title, needed)

    rendered = [lookup.get((str(d), str(i)), ("", False))
                for d, i in zip(data["result_domain"], data["item_id"])]
    data["item_text"] = [t for t, _ in rendered]
    data["source_thin"] = [thin for _, thin in rendered]
    missing = int((data["item_text"] == "").sum())
    if missing:
        print(f"  [warn] 원문을 못 찾은 행 {missing:,}건 — 판정에서 제외")
    data = data[data["item_text"] != ""].reset_index(drop=True)

    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    # 서명을 키에도 넣는다 — 캐시 파일을 실수로 합쳐도 조건이 다른 등급끼리 섞이지 않는다.
    data["judge_key"] = [
        f"{sig}|{d}|{i}|{q}"
        for d, i, q in zip(data["result_domain"], data["item_id"], data["query"])
    ]
    todo = data[~data["judge_key"].isin(cache)].drop_duplicates("judge_key")
    print(f"판정 대상 {len(todo):,}건 (캐시 {len(cache):,}건 재사용)")

    if len(todo):
        from scripts.vllm_runner import VLLMRunner, chunks
        # 블라인드: 실행 라벨을 프롬프트에 넣지 않고, 순서도 섞어 배치 구성이 실행별로
        # 쏠리지 않게 한다(설계 (1)).
        todo = todo.sample(frac=1.0, random_state=42).reset_index(drop=True)
        template = build_prompt(args.mask_title)
        runner = VLLMRunner(args.model_id, max_new_tokens=4)

        done = 0
        os.makedirs(CACHE_DIR, exist_ok=True)
        for batch in chunks(list(todo.itertuples()), args.batch):
            items = []
            for r in batch:
                noun, noun_title = NOUNS[r.result_domain]
                items.append((template.format(noun=noun, noun_title=noun_title,
                                              query=r.query, item=r.item_text), None))
            for r, raw in zip(batch, runner.generate(items)):
                g = parse_grade(raw)
                if g:
                    cache[r.judge_key] = g
            done += len(batch)
            json.dump(cache, open(cache_path, "w"))
            print(f"  진행 {done:,}/{len(todo):,}", flush=True)

    data["grade"] = data["judge_key"].map(cache)
    data["value"] = data["grade"].map(GRADE_VALUE)
    judged = data[data["value"].notna()]
    # 주 지표는 자유 서술이 있는 행으로만 낸다. 결손 행은 빼지 않고 아래에서 따로 센다.
    main_rows = judged[~judged["source_thin"]]
    runs = [r for r in run_order if (judged["run_label"] == r).any()]

    lines = [f"LLM 관련성 판정 리포트 ({args.model_id})",
             f"조건: 제목 {'가림' if args.mask_title else '노출'} / 언어 {args.lang} / 서명 {sig}",
             "=" * 68, ""]

    # 판정기 신뢰도 — 6월 행에 남은 사람/Claude 라벨과 얼마나 맞는가. 낮으면 이하 수치를
    # 믿을 수 없으므로 맨 앞에 둔다. 프롬프트를 바꿀 때마다 다시 재야 하는 값이다.
    if "Validity" in judged.columns:
        v = judged[judged["Validity"].notna()]
        if len(v):
            agree = (v["Validity"].str.strip() == v["grade"]).mean()
            adj = (v["Validity"].map(GRADE_VALUE) - v["value"]).abs().le(1).mean()
            lines += ["## 0. 판정기 신뢰도 (6월 사람/Claude 라벨 대비)",
                      f"  완전일치 {agree:.1%} / 인접일치(±1등급) {adj:.1%} / 대상 {len(v):,}행",
                      "  ※ 완전일치가 낮으면 이하 절대값보다 실행 간 '차이'만 보아야 한다.", ""]

    # 설명 결손 노출률 — 실행의 성질이지 표본의 결함이 아니다. music의 9.1%가 여기 걸리고
    # 실행별 편차가 크다(2026-09-08 실측: meanpool 17.3% vs centered 6.7%).
    lines += ["## 1. 설명 결손 노출률 (자유 서술이 없는 아이템을 몇 위에 올렸나)",
              f"  {'실행':14s} {'결손행':>7s} {'전체행':>7s} {'전체대비':>9s} "
              f"{'music대비':>10s} {'상위3위내':>9s}"]
    for run in runs:
        g = judged[judged["run_label"] == run]
        mus = g[g["result_domain"] == "music"]
        top3 = g[g["rank"] <= 3] if "rank" in g.columns else g.iloc[0:0]
        thin = int(g["source_thin"].sum())
        lines.append(
            f"  {run:14s} {thin:>7,} {len(g):>7,} {g['source_thin'].mean():>9.1%} "
            f"{(mus['source_thin'].mean() if len(mus) else float('nan')):>10.1%} "
            f"{(top3['source_thin'].mean() if len(top3) else float('nan')):>9.1%}")
    lines += ["  ※ 이 행들도 판정에는 들어간다. 제외하면 실행마다 빠지는 행 수가 달라",
              "     서로 다른 표본의 평균을 비교하게 된다. 아래 2절부터는 결손을 뺀 값이다.", ""]

    lines += _run_table(main_rows, runs, "## 2. 실행별 종합 (결손 제외 — 주 지표)")
    lines += [""] + _run_table(judged, runs, "  [참고: 결손 포함 전체]")

    # 확장 쿼리(pair_id 5~10)는 6월에 없다. 섞어서 평균 내면 baseline 대비 수치가 무의미해지므로
    # 같은 쿼리 집합끼리만 비교할 수 있도록 분리해 둔다.
    if "pair_id" in main_rows.columns and (main_rows["pair_id"] > 4).any():
        base = main_rows[main_rows["pair_id"] <= 4]
        lines += ["", "  [6월과 공통인 쿼리(pair_id 1~4)로 한정]"]
        for run in runs:
            g = base[base["run_label"] == run]
            if not len(g):
                continue
            se = g["value"].std(ddof=1) / (len(g) ** 0.5)
            lines.append(f"  {run:14s} {len(g):>6,} {g['value'].mean():>7.3f} {se:>6.3f}")

    lines += ["", "## 3. 스타일별 avg_score (핵심 — poet이 6월 최하였다)"]
    piv = main_rows.pivot_table(index="style", columns="run_label", values="value", aggfunc="mean")
    cols = [r for r in runs if r in piv.columns]
    # 마지막 두 실행의 차이를 표준오차로 나눠 노이즈와 구분한다. 스타일당 표본이 적으면
    # 0.1 수준의 차이는 판정할 수 없다 — 8/13 실행에서 poet +0.138이 1.5 SE였다.
    lines.append("  " + f"{'스타일':16s}" + "".join(f"{r:>14s}" for r in cols) +
                 f"{'변화':>9s}{'SE배수':>8s}")
    for style, row in piv.iterrows():
        cells = "".join(f"{row[r]:>14.3f}" for r in cols)
        tail = ""
        if len(cols) > 1:
            # main_rows["style"]로 써야 한다 — .style은 pandas의 Styler 속성이라
            # 컬럼이 아니라 Styler 객체가 잡히고, 비교가 조용히 전부 False가 된다.
            a = main_rows[(main_rows["style"] == style) & (main_rows["run_label"] == cols[-2])]["value"]
            b = main_rows[(main_rows["style"] == style) & (main_rows["run_label"] == cols[-1])]["value"]
            delta = b.mean() - a.mean()
            se = ((a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 0.5
                  if len(a) > 1 and len(b) > 1 else 0.0)
            tail = f"{delta:>+9.3f}{(abs(delta) / se if se else 0):>8.1f}"
        lines.append("  " + f"{style:16s}" + cells + tail)
    lines.append("  ※ SE배수 2 이상이면 유의, 1 미만이면 노이즈로 본다 (마지막 두 실행 기준)")

    lines += ["", "## 4. 도메인별 avg_score"]
    piv2 = main_rows.pivot_table(index="domain_filter", columns="run_label", values="value", aggfunc="mean")
    cols2 = [r for r in runs if r in piv2.columns]
    lines.append("  " + f"{'도메인':16s}" + "".join(f"{r:>10s}" for r in cols2) + f"{'변화':>9s}")
    for dom, row in piv2.iterrows():
        delta = row[cols2[-1]] - row[cols2[0]] if len(cols2) > 1 else float("nan")
        lines.append("  " + f"{dom:16s}" + "".join(f"{row[r]:>10.3f}" for r in cols2) +
                     (f"{delta:>+9.3f}" if len(cols2) > 1 else ""))

    lines += ["", "## 5. 쿼리별 (신규 실행 기준 오름차순 = 개선 우선순위)"]
    piv3 = main_rows.pivot_table(index=["style", "query"], columns="run_label", values="value", aggfunc="mean")
    cols3 = [r for r in runs if r in piv3.columns]
    for (style, query), row in piv3.sort_values(cols3[-1]).iterrows():
        vals = "".join(f"{row[r]:>8.2f}" if pd.notna(row[r]) else f"{'-':>8s}" for r in cols3)
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
