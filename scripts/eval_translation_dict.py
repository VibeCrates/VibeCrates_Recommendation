"""한국어 검색어 번역 품질을 잰다 — opus-mt(이전 서빙) / Gemma 27B / 사전+opus-mt(현재 서빙).

무엇을 재는가:
  1. 로그 530종 (experiments/translation_ko530_compare.csv)
     서버 로그에서 백엔드가 보낸 한국어 검색어 530종과 그때의 opus-mt 번역, 같은 검색어의
     Gemma 27B 번역(beast, 2026-10-12, build_translation_dict.PROMPT와 같은 프롬프트)이다. 검색어를 단위("주말 오후", "추리소설")로 쪼개고,
     단위마다 번역에 있어야 할 영어 표현(GLOSSARY)이 빠졌으면 "핵심어 손실"로 센다.
     GLOSSARY는 관대하다 — 공포를 "Fear"로 옮겨도 통과다. 그래서 손실률은 하한이다.
     **이 530종은 사전 어휘를 고른 출처이므로 사전 쪽 수치는 표본 안 성적이다.**
  2. 표본 밖 적중률 — 사전을 만들 때 보지 않은 한국어 검색어에서 사전이 몇 개를 풀어내는지.
     평가 쿼리 40개(문장형)와 학습 쿼리 번들의 query_ko(있으면) 무작위 2,000개.
     사전이 못 푼 것은 opus-mt로 가므로, 적중률이 낮다는 것은 "나빠진다"가 아니라
     "이전과 같다"는 뜻이다.

사용:
  python scripts/eval_translation_dict.py
"""
import collections
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.api.translation import DictTranslator

LOG_CSV = "experiments/translation_ko530_compare.csv"
EVAL_CSV = "experiments/eval_lang_ext_qlora_off.csv"
BUNDLE_KO = "dist/query_vectors_20261009_ko/movie_query_vectors.csv"
OUT_TXT = "experiments/translation_dict_eval.txt"

# 한국어 단위 → 번역에 있어야 하는 영어 표현(정규식, 모두 만족해야 통과)
GLOSSARY = {
 "주말 오후":[r"weekend",r"afternoon"], "공부할 때":[r"stud"], "겨울 아침":[r"winter",r"morning"],
 "잠들기 전":[r"before (bed|sleep|you go to sleep)|bedtime"], "운동할 때":[r"workout|exercis|working out|gym"],
 "새벽 감성":[r"dawn|late.night|early morning|small hours"], "비 오는 날":[r"rain"],
 "여름 밤":[r"summer",r"night"], "비 오는 오후의 조용한 시간":[r"rain",r"afternoon",r"quiet"],
 "겨울 밤 기차역":[r"winter",r"night",r"train station"], "듣기 좋은":[], "잠 안 오는 밤":[r"sleepless|can.t sleep|not sleeping|insomnia"],
 "숲":[r"forest|wood"], "심리학":[r"psycholog"], "재즈":[r"jazz"], "보사노바":[r"bossa ?nova"],
 "판타지":[r"fantas"], "우정":[r"friend"], "코미디":[r"comed"], "강아지":[r"pupp|dog"],
 "복수":[r"reveng|vengeance|aveng"], "바다":[r"\bsea|ocean|beach|seaside"], "첫사랑":[r"first love"],
 "시집":[r"poetry|poems"], "여행":[r"trav|trip|journey"], "생존":[r"surviv"], "스릴러":[r"thriller"],
 "전쟁":[r"\bwar\b"], "시간여행":[r"time travel"], "다큐멘터리":[r"documentar"], "클래식":[r"classic"],
 "시티팝":[r"city ?pop"], "로맨스":[r"romanc|romantic"], "인디":[r"indie"], "경제":[r"econom"],
 "기차":[r"train"], "힙합":[r"hip.?hop"], "로파이":[r"lo.?fi"],
 "이별":[r"break.?up|farewell|separation|parting|heartbreak"], "자기계발":[r"self\W*(help|improvement|development)"],
 "발라드":[r"ballad"], "마법":[r"magic"], "고양이":[r"\bcat"], "철학":[r"philosoph"],
 "범죄":[r"crime|criminal"], "성장":[r"coming.of.age|growing up|growth|grow"], "역사":[r"histor"],
 "가족":[r"family"], "록":[r"\brock"], "에세이":[r"essay"], "도시":[r"city|urban"],
 "공포":[r"horror|fear|scary|terror"], "드라이브":[r"driv|road trip"],
 "퇴근길":[r"commute|way home|home from work|after work|going home|ride home"], "로봇":[r"robot"],
 "추리소설":[r"mystery|detective",r"novel|fiction|stor"], "요리":[r"cook|cuisine|food"], "사막":[r"desert"],
 "피아노":[r"piano"], "모험":[r"adventur"], "애니메이션":[r"anim"], "인공지능":[r"\bAI\b|artificial intelligence"],
 "우주":[r"space|universe|cosmos"], "사랑":[r"love"], "재미":[r"\bfun"], "화가":[r"painter|artist"],
 "이유":[r"reason"], "잔잔한":[r"calm|quiet|mellow|gentle|serene"], "영화":[r"movie|film"],
 "노래":[r"song"], "음악":[r"music"], "하늘":[r"\bsky"], "매정":[r"cold|heartless|callous|unfeeling"],
 "테스트":[r"test"],
}
UNITS = sorted(GLOSSARY, key=len, reverse=True)


def parse(q: str) -> list[str] | None:
    rest, got = q, []
    while rest.strip():
        rest = rest.strip()
        m = re.match(r"\d+", rest)
        if m:
            got.append(m.group())
            rest = rest[m.end():]
            continue
        for u in UNITS:
            if rest.startswith(u):
                got.append(u)
                rest = rest[len(u):]
                break
        else:
            return None
    return got


def missing(units: list[str], text: str) -> list[str]:
    miss = []
    for u in units:
        pats = [re.escape(u)] if u.isdigit() else GLOSSARY[u]
        if not all(re.search(p, str(text), re.I) for p in pats):
            miss.append(u)
    return miss


def main():
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    d = DictTranslator()

    # ── 1. 로그 530종 ────────────────────────────────────────────────────
    df = pd.read_csv(LOG_CSV)
    df["dict"] = df["query"].map(d.translate)
    df["served"] = df["dict"].fillna(df["opus"])
    df["units"] = df["query"].map(parse)
    sc = df[df["units"].notna()].copy()
    say(f"[1] 로그 검색어 {len(df)}종 — 채점 가능 {len(sc)} (단위로 못 쪼갠 입력 "
        f"{len(df) - len(sc)}개 제외: {list(df[df['units'].isna()]['query'])})")
    say(f"    사전 적중 {df['dict'].notna().sum()}/{len(df)} (표본 안 — 이 검색어들에서 어휘를 골랐다)")
    say(f"    {'번역':22s} 핵심어 손실 쿼리   요청 수 가중")
    for col, name in (("opus", "opus-mt (이전 서빙)"), ("gemma", "Gemma 27B"),
                      ("served", "사전+opus (현재 서빙)")):
        sc[f"{col}_miss"] = [missing(u, t) for u, t in zip(sc["units"], sc[col])]
        bad = sc[sc[f"{col}_miss"].map(len) > 0]
        say(f"    {name:22s} {len(bad):3d}/{len(sc)} = {len(bad) / len(sc):5.1%}   "
            f"{bad['n'].sum() / sc['n'].sum():5.1%}")
    c = collections.Counter(u for m in sc["served_miss"] for u in m)
    say(f"    현재 서빙이 잃는 단위: {c.most_common()}")
    for r in sc[sc["served_miss"].map(len) > 0].itertuples():
        say(f"      {r.query} → {r.served}")

    # ── 2. 표본 밖 적중률 ─────────────────────────────────────────────────
    say()
    ev = pd.read_csv(EVAL_CSV)
    ko = ev.loc[ev["lang"] == "ko", "query"].drop_duplicates().tolist()
    hits = [(q, d.translate(q)) for q in ko]
    hits = [(q, t) for q, t in hits if t]
    say(f"[2] 평가 쿼리 40개(문장형) 사전 적중 {len(hits)}/{len(ko)}")
    for q, t in hits:
        say(f"      {q} → {t}")
    if os.path.exists(BUNDLE_KO):
        b = pd.read_csv(BUNDLE_KO, usecols=["query", "query_ko"]).dropna()
        b = b.sample(n=min(2000, len(b)), random_state=20261012)
        b["dict"] = b["query_ko"].map(d.translate)
        h = b[b["dict"].notna()]
        say(f"    학습 쿼리 query_ko 무작위 {len(b)}개 사전 적중 {len(h)} ({len(h) / len(b):.1%})")
        for r in h.head(10).itertuples():
            say(f"      {r.query_ko} → {r.dict}   (원문 영어: {r.query})")
    else:
        say(f"    ({BUNDLE_KO} 없음 — query_ko 적중률은 건너뜀)")

    df.drop(columns=["units"]).to_csv(LOG_CSV, index=False)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"→ {OUT_TXT}, {LOG_CSV}에 dict·served 열 추가")


if __name__ == "__main__":
    main()
