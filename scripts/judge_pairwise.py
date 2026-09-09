"""
두 실행의 검색 결과를 쌍대 비교로 판정한다 (절대 등급 o/a/x의 대안).

왜 필요한가:
  절대 등급은 눈금 보정이 흔들린다. 같은 판정기가 'a' 비율을 0.6%에서 61%로 널뛰었고,
  6월 사람 라벨 대비 완전일치는 38~42%다. 그 저울로 잰 미결 넷이 9/8 재판정에서도
  0.0~1.2 SE로 전부 안 갈렸다(worklog 3-6). 쌍대 비교는 눈금을 요구하지 않는다 —
  "어느 쪽이 더 맞는가"만 물으므로 판정기가 절대 기준을 일관되게 유지할 필요가 없고,
  같은 판정 예산에서 더 작은 차이를 잡는다.

설계:

  (1) 단위는 아이템이 아니라 **리스트**다. (쿼리, 도메인필터)마다 두 실행의 top-k를
      통째로 놓고 "어느 목록이 이 쿼리에 더 맞는가"를 묻는다. 실행 비교가 목적이므로
      아이템 단위로 쪼개면 두 실행이 서로 다른 아이템을 뽑았다는 사실을 잃는다.

  (2) 위치 편향은 **양방향 질문으로 잰다.** A/B를 바꿔 두 번 묻고, 두 답이 같은 실행을
      가리킬 때만 승패로 센다. 어긋나면 무승부로 처리하고 일관성률을 따로 보고한다.
      앞에 놓인 쪽을 고른 비율(위치 편향)도 함께 낸다.

  (3) 제목은 절대 판정과 같은 방식으로 가린다 — `judge_eval_relevance.render_item`을
      그대로 재사용한다. 렌더가 바뀌면 서명이 바뀌어 캐시가 갈리는 것도 같다.

  (4) 실행 라벨은 프롬프트에 넣지 않는다(블라인드). 캐시 키에만 남는다.

  --validate-june은 6월 사람 라벨로 판정기 자체를 잰다. 같은 쿼리에서 사람이 o를 준
  아이템과 x를 준 아이템을 짝지어 물어, 판정기가 o 쪽을 고르는 비율과 위치 편향을 낸다.
  사람 라벨 400건(1-3)을 새로 만들지 않고 쓸 수 있는 유일한 외부 앵커다.

사용:
  python scripts/judge_pairwise.py --lang en \
      --a qlora_off:experiments/eval_lang_ext_qlora_off.csv \
      --b qlora_on:experiments/eval_lang_ext_qlora_on.csv \
      --out experiments/pair_en_qlora.txt
  python scripts/judge_pairwise.py --validate-june --out experiments/pair_june_validation.txt
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random

import pandas as pd

from scripts.judge_eval_relevance import (
    CACHE_DIR,
    NOUNS,
    build_item_lookup,
    render_item,
    scrub_title,
    _title_pattern,
)

# 리스트 프롬프트는 아이템 20개가 한 번에 들어가므로 본문을 줄인다. max_model_len 8192에
# 여유를 두려면 아이템당 이 정도가 상한이다(20개 × 320자 ≈ 1,800토큰).
ITEM_CHARS = 280

LIST_PROMPT = (
    "[Task]\n"
    "A user searched with an evocative, vibe-based query. Two systems each returned a list "
    "of {noun}s. Decide which list is the better set of results for that query.\n\n"
    "[Query]\n{query}\n\n"
    "[List 1]\n{list1}\n\n"
    "[List 2]\n{list2}\n\n"
    "[Rules]\n"
    "- Titles are withheld on purpose. Judge from the remaining fields; do not penalise the "
    "missing title and do not try to guess what the works are called.\n"
    "- Judge the WORKS themselves, not how well they are described. A thin entry is not a "
    "reason to reject a list on its own.\n"
    "- Vibe queries are metaphorical. A work matches if its atmosphere or theme resonates, "
    "even if no words overlap literally.\n"
    "- Weigh the list as a whole. One strong hit does not outweigh several misses.\n"
    "- Do not favour a list for being first or longer.\n"
    "- You MUST pick one. If the two lists look close, choose the one whose weakest "
    "entries are less off-target.\n\n"
    "[Output]\nExactly one character: 1 or 2. No explanation."
)

ITEM_PROMPT = (
    "[Task]\n"
    "A user searched with an evocative, vibe-based query. Two {noun}s were retrieved. "
    "Decide which one is the better match for that query.\n\n"
    "[Query]\n{query}\n\n"
    "[Candidate 1]\n{item1}\n\n"
    "[Candidate 2]\n{item2}\n\n"
    "[Rules]\n"
    "- Titles are withheld on purpose. Judge from the remaining fields.\n"
    "- Judge the WORK itself, not how well it is described.\n"
    "- Vibe queries are metaphorical; atmosphere and theme count more than word overlap.\n"
    "- Do not favour a candidate for being first.\n"
    "- You MUST pick one. If the two look close, choose the one whose connection to the "
    "query is less of a stretch.\n\n"
    "[Output]\nExactly one character: 1 or 2. No explanation."
)


def prompt_signature() -> str:
    """프롬프트와 렌더 방식의 해시. 절대 판정과 같은 이유로 캐시 키에 넣는다 —
    프롬프트를 고치고 캐시를 그대로 쓰면 옛 답이 새 조건의 결과인 척 집계된다."""
    src = "".join([
        LIST_PROMPT, ITEM_PROMPT, str(ITEM_CHARS),
        inspect.getsource(render_item),
        inspect.getsource(scrub_title),
        inspect.getsource(_title_pattern),
    ])
    return hashlib.sha1(src.encode()).hexdigest()[:8]


def parse_choice(raw: str) -> str | None:
    """'1' / '2' / 'T' 중 하나. 판정기가 문장으로 답하면 첫 토큰만 본다."""
    s = str(raw).strip().lower()
    for ch in s[:6]:
        if ch in "12":
            return ch
        if ch == "t":
            return "T"
    return None


def _fmt_list(texts: list[str]) -> str:
    out = []
    for i, t in enumerate(texts, 1):
        body = t.replace("\n", " / ")
        out.append(f"{i}. {body[:ITEM_CHARS]}")
    return "\n".join(out)


# ── 비교 단위 만들기 ──────────────────────────────────────────────────────────

def load_run(path: str, lang: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if lang != "all" and "lang" in df.columns:
        df = df[df["lang"] == lang]
    return df


def build_item_units(a: pd.DataFrame, b: pd.DataFrame, lookup: dict, topk: int,
                     per_unit: int) -> list[dict]:
    """(쿼리, 도메인필터)마다 **두 실행이 서로 다르게 뽑은** 아이템을 순위 순으로 짝짓는다.

    공통으로 뽑은 아이템은 건드리지 않는다. 무승부 선택지를 없앴으므로 같은 아이템끼리
    물으면 잡음만 만든다. 실행 차이는 어차피 서로 다르게 뽑은 부분에서만 나온다.
    """
    units = []
    # query_id로 짝지으면 안 된다 — 실행마다 접미사가 붙는 경우가 있다(P1_KO_opus vs
    # P1_KO_qwen). (스타일, 쿼리번호, 도메인필터)는 실행이 달라도 같은 쿼리를 가리킨다.
    keys = ["style", "pair_id", "domain_filter"]
    bi = {k: g for k, g in b.groupby(keys)}
    for k, ga in a.groupby(keys):
        gb = bi.get(k)
        if gb is None:
            continue
        ga = ga.sort_values("rank").head(topk)
        gb = gb.sort_values("rank").head(topk)
        sa = set(zip(ga["result_domain"].astype(str), ga["item_id"].astype(str)))
        sb = set(zip(gb["result_domain"].astype(str), gb["item_id"].astype(str)))
        onlya = [r for r in ga.itertuples()
                 if (str(r.result_domain), str(r.item_id)) not in sb]
        onlyb = [r for r in gb.itertuples()
                 if (str(r.result_domain), str(r.item_id)) not in sa]
        for ra, rb in list(zip(onlya, onlyb))[:per_unit]:
            ta = lookup.get((str(ra.result_domain), str(ra.item_id)), ("", False))[0]
            tb = lookup.get((str(rb.result_domain), str(rb.item_id)), ("", False))[0]
            if not ta or not tb:
                continue
            noun = (NOUNS[ra.result_domain][0]
                    if ra.result_domain == rb.result_domain else "item")
            units.append({
                "uid": f"{k[0]}|{k[1]}|{k[2]}|{ra.item_id}|{rb.item_id}",
                "query_id": f"{k[0]}{k[1]}", "domain_filter": k[2],
                "style": ra.style, "query": ra.query, "noun": noun,
                "a": [ta], "b": [tb],
            })
    return units


def build_units(a: pd.DataFrame, b: pd.DataFrame, lookup: dict, topk: int) -> list[dict]:
    """(query_id, domain_filter)마다 두 실행의 top-k 리스트 한 쌍."""
    units = []
    keys = ["style", "pair_id", "domain_filter"]
    bi = {k: g for k, g in b.groupby(keys)}
    for k, ga in a.groupby(keys):
        gb = bi.get(k)
        if gb is None:
            continue
        ga = ga.sort_values("rank").head(topk)
        gb = gb.sort_values("rank").head(topk)
        ta = [lookup.get((str(d), str(i)), ("", False))[0]
              for d, i in zip(ga["result_domain"], ga["item_id"])]
        tb = [lookup.get((str(d), str(i)), ("", False))[0]
              for d, i in zip(gb["result_domain"], gb["item_id"])]
        ta = [t for t in ta if t]
        tb = [t for t in tb if t]
        if not ta or not tb:
            continue
        # 도메인필터가 'all'이면 세 도메인이 섞이므로 명사를 일반화한다.
        doms = set(ga["result_domain"]) | set(gb["result_domain"])
        noun = NOUNS[doms.pop()][0] if len(doms) == 1 else "item"
        units.append({
            "uid": f"{k[0]}|{k[1]}|{k[2]}",
            "query_id": f"{k[0]}{k[1]}", "domain_filter": k[2],
            "style": ga.iloc[0]["style"], "query": ga.iloc[0]["query"],
            "noun": noun, "a": ta, "b": tb,
        })
    return units


def build_june_pairs(lookup: dict, limit: int, seed: int, require_body: bool = True) -> list[dict]:
    """6월 사람 라벨에서 (같은 쿼리의 o 아이템, x 아이템) 쌍을 만든다.

    자유 서술이 없는 아이템은 기본으로 제외한다. 6월 book 행의 94.2%가 결손이라
    가림 조건에서 "Author: Jean-Paul Sartre / Category: …" 대 "Genre: Documentary"를
    비교하게 되는데, 그것은 판정기가 사르트르를 아는지 재는 것이지 판정 능력을 재는
    것이 아니다(worklog 3-5 (마)).
    """
    v = pd.read_csv("experiments/eval_lang_20260618_en_valids.csv")
    v = v[v["Validity"].notna()].copy()
    v["Validity"] = v["Validity"].str.strip()
    pairs = []
    for q, g in v.groupby("query"):
        pos = g[g["Validity"] == "o"]
        neg = g[g["Validity"] == "x"]
        for _, p in pos.iterrows():
            for _, n in neg.iterrows():
                tp, thin_p = lookup.get((str(p["result_domain"]), str(p["item_id"])), ("", True))
                tn, thin_n = lookup.get((str(n["result_domain"]), str(n["item_id"])), ("", True))
                if not tp or not tn:
                    continue
                if require_body and (thin_p or thin_n):
                    continue
                pairs.append({
                    "uid": f"{p['item_id']}|{n['item_id']}|{q}",
                    "query": q, "style": p["style"], "domain_filter": p["domain_filter"],
                    "noun": NOUNS[p["result_domain"]][0]
                            if p["result_domain"] == n["result_domain"] else "item",
                    "a": [tp], "b": [tn],   # a = 사람이 o를 준 쪽
                })
    random.Random(seed).shuffle(pairs)
    return pairs[:limit]


# ── 판정 ─────────────────────────────────────────────────────────────────────

def make_prompt(unit: dict, first: str, item_mode: bool) -> str:
    """first가 'a'면 A를 1번 자리에 놓는다."""
    x, y = (unit["a"], unit["b"]) if first == "a" else (unit["b"], unit["a"])
    if item_mode:
        return ITEM_PROMPT.format(noun=unit["noun"], query=unit["query"],
                                  item1=x[0][:ITEM_CHARS * 2].replace("\n", " / "),
                                  item2=y[0][:ITEM_CHARS * 2].replace("\n", " / "))
    return LIST_PROMPT.format(noun=unit["noun"], query=unit["query"],
                              list1=_fmt_list(x), list2=_fmt_list(y))


def verdict(choice: str, first: str) -> str:
    """판정기의 '1/2/T'를 실행 라벨('a'/'b'/'T')로 되돌린다."""
    if choice == "T":
        return "T"
    if first == "a":
        return "a" if choice == "1" else "b"
    return "b" if choice == "1" else "a"


def sign_test(w: int, l: int) -> float:
    """양측 부호검정 p값. 승패만 세고 무승부는 버린다(scipy 의존을 만들지 않는다)."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", help="라벨:CSV — 기준 실행")
    ap.add_argument("--b", help="라벨:CSV — 비교 실행")
    ap.add_argument("--lang", default="en", choices=["en", "ko", "all"])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--unit", default="item", choices=["item", "list"],
                    help="item=두 실행이 다르게 뽑은 아이템끼리 (권장), list=top-k 목록 통째로")
    ap.add_argument("--per-unit", type=int, default=5,
                    help="unit=item일 때 (쿼리,도메인)당 만들 쌍의 최대 개수")
    ap.add_argument("--validate-june", action="store_true",
                    help="6월 사람 라벨의 o/x 쌍으로 판정기 자체를 잰다")
    ap.add_argument("--june-limit", type=int, default=200)
    ap.add_argument("--june-allow-thin", action="store_true",
                    help="자유 서술이 없는 아이템도 검증 쌍에 넣는다 (기본은 제외)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dry-run", type=int, default=0, help="프롬프트 N개만 찍고 끝낸다")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sig = prompt_signature()
    item_mode = args.validate_june or args.unit == "item"
    if not args.validate_june and not (args.a and args.b):
        ap.error("--a와 --b가 필요하다 (또는 --validate-june)")

    # item_mode는 프롬프트 모양(아이템 대 아이템)만 정한다. 데이터를 어디서 읽을지는
    # --validate-june만이 정한다 — 둘을 한 조건으로 묶으면 --unit item이 6월 파일을 읽는다.
    if args.validate_june:
        label_a, label_b = "사람 o", "사람 x"
        need_frames = [pd.read_csv("experiments/eval_lang_20260618_en_valids.csv")]
    else:
        label_a, path_a = args.a.split(":", 1)
        label_b, path_b = args.b.split(":", 1)
        fa, fb = load_run(path_a, args.lang), load_run(path_b, args.lang)
        need_frames = [fa, fb]
        print(f"[{label_a}] {len(fa):,}행 / [{label_b}] {len(fb):,}행 (lang={args.lang})")

    needed = set()
    for f in need_frames:
        needed |= {(str(d), str(i)) for d, i in zip(f["result_domain"], f["item_id"])}
    print("아이템 원문 조회 준비...")
    lookup = build_item_lookup(True, needed)

    if args.validate_june:
        units = build_june_pairs(lookup, args.june_limit, args.seed, not args.june_allow_thin)
    elif args.unit == "item":
        units = build_item_units(fa, fb, lookup, args.topk, args.per_unit)
    else:
        units = build_units(fa, fb, lookup, args.topk)
    print(f"비교 단위 {len(units):,}개 × 양방향 2회 = 질문 {len(units)*2:,}건 (서명 {sig})")

    if args.dry_run:
        for u in units[:args.dry_run]:
            print("=" * 70)
            print(make_prompt(u, "a", item_mode))
        return

    cache_path = os.path.join(CACHE_DIR, f"judge_cache_pair_{sig}.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [(u, first) for u in units for first in ("a", "b")
            if f"{sig}|{label_a}|{label_b}|{u['uid']}|{first}" not in cache]
    print(f"판정 대상 {len(todo):,}건 (캐시 {len(cache):,}건 재사용)")

    if todo:
        from scripts.vllm_runner import VLLMRunner, chunks
        runner = VLLMRunner(args.model_id, max_new_tokens=4)
        os.makedirs(CACHE_DIR, exist_ok=True)
        done = 0
        for batch in chunks(todo, args.batch):
            items = [(make_prompt(u, first, item_mode), None) for u, first in batch]
            for (u, first), raw in zip(batch, runner.generate(items)):
                c = parse_choice(raw)
                if c:
                    cache[f"{sig}|{label_a}|{label_b}|{u['uid']}|{first}"] = c
            done += len(batch)
            json.dump(cache, open(cache_path, "w"))
            print(f"  진행 {done:,}/{len(todo):,}", flush=True)

    # ── 집계 ─────────────────────────────────────────────────────────────────
    rows = []
    first_pick = 0      # 앞에 놓인 쪽을 고른 횟수 (위치 편향)
    decisive = 0        # 1/2로 답한 질문 수. T는 편향의 분모가 아니다
    ties = 0
    for u in units:
        vs = {}
        for first in ("a", "b"):
            c = cache.get(f"{sig}|{label_a}|{label_b}|{u['uid']}|{first}")
            if c is None:
                break
            if c in "12":
                decisive += 1
                first_pick += (c == "1")
            else:
                ties += 1
            vs[first] = verdict(c, first)
        if len(vs) < 2:
            continue
        agree = vs["a"] == vs["b"]
        # 두 방향이 같은 실행을 가리킬 때만 승패로 센다. 어긋나면 판정기가 위치에
        # 흔들린 것이므로 무승부로 처리한다.
        res = vs["a"] if agree and vs["a"] != "T" else "T"
        rows.append({"style": u["style"], "domain_filter": u["domain_filter"],
                     "query": u["query"], "res": res, "agree": agree,
                     "raw_a": vs["a"], "raw_b": vs["b"]})
    df = pd.DataFrame(rows)

    L = [f"쌍대 비교 리포트 ({args.model_id})",
         f"조건: 제목 가림 / 단위 {'아이템(6월 사람 라벨 검증)' if args.validate_june else ('아이템(실행 간 차집합)' if args.unit == 'item' else f'리스트 top-{args.topk}')}"
         f" / 언어 {args.lang if not item_mode else 'en'} / 서명 {sig}",
         f"A = {label_a} / B = {label_b}", "=" * 68, ""]

    if not len(df):
        L.append("집계할 단위가 없다.")
    else:
        w = int((df.res == "b").sum()); l = int((df.res == "a").sum()); t = int((df.res == "T").sum())
        L += ["## 0. 판정기 자체 점검",
              f"  양방향 일관성 {df.agree.mean():.1%} (두 번 물어 같은 쪽을 고른 비율)",
              f"  위치 편향 {first_pick/max(decisive,1):.1%} (결정한 질문 중 앞자리를 고른 비율, 50%가 중립)",
              f"  판단 회피 {ties/max(decisive+ties,1):.1%} (선택지에 없는 무승부로 답한 비율)",
              "  ※ 위치 편향이 40~60%를 벗어나거나 일관성이 낮으면 이하 승패를 믿을 수 없다.", "",
              "## 1. 종합",
              f"  B 승 {w} / A 승 {l} / 불일치(위치에 흔들림) {t}  (단위 {len(df)}개)",
              f"  B 승률 {w/max(w+l,1):.1%} (승패만), 부호검정 p = {sign_test(w,l):.4f}",
              "  ※ p < 0.05면 두 실행에 실제 차이가 있다고 본다.", ""]

        for col, title in (("style", "2. 스타일별"), ("domain_filter", "3. 도메인필터별")):
            L.append(f"## {title}")
            L.append(f"  {'':16s}{'B승':>5s}{'A승':>5s}{'무':>5s}{'B승률':>9s}{'p':>9s}")
            for k, g in df.groupby(col):
                gw = int((g.res == "b").sum()); gl = int((g.res == "a").sum())
                gt = int((g.res == "T").sum())
                L.append(f"  {str(k):16s}{gw:>5d}{gl:>5d}{gt:>5d}"
                         f"{gw/max(gw+gl,1):>9.1%}{sign_test(gw,gl):>9.4f}")
            L.append("")

    report = "\n".join(L)
    print("\n" + report)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(report + "\n")
    if len(df):
        df.to_csv(args.out.replace(".txt", "_rows.csv"), index=False)
    print(f"\n리포트 → {args.out}")


if __name__ == "__main__":
    main()
