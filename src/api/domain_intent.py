"""검색어가 도메인을 명시했는지 읽는다 (2026-10-12).

백엔드는 검색어만 보낸다({"text": "..."}). 사용자가 "잔잔한 영화", "비 오는 날 듣기 좋은 노래",
"추리소설"처럼 원하는 작품 종류를 밝혔으면 백엔드가 그 도메인으로 검색을 제한할 수 있게
응답에 `domain`을 실어 준다. 밝히지 않았으면 "all"이다.

원칙 — **확실할 때만 제한한다.**
  놓치면(all) 결과에 다른 도메인이 섞일 뿐 원하는 도메인도 나온다. 잘못 제한하면 원하는
  결과가 사라진다. 그래서:
  - 작품 종류를 가리키는 명사(영화·노래·책·소설·시집…)만 단서로 본다.
  - 장르는 단서가 아니다. 재즈·공포·다큐멘터리는 여러 도메인에 걸치고, 발라드·힙합·로파이처럼
    사실상 한 도메인뿐인 장르도 사용자가 종류를 밝힌 것은 아니므로 넣지 않는다(10/12 결정).
  - 동사(듣기 좋은, 읽을)도 넣지 않는다(10/12 결정).
  - 애매하면 all.

원문에서 읽는다 — 번역문이 아니다:
  번역기가 도메인 명사를 자주 잃는다(시집 → "Indulgent", 추리소설 → "The Story of Reason").
  번역 사전을 넣은 뒤에도 사전에 없는 말이 섞이면 opus-mt로 가므로, 한국어는 한국어로 읽는다.

단서가 여럿일 때 — 수식받는 쪽(핵심 명사)을 고른다:
  - "영화 같은 노래", "책처럼 읽히는 앨범": 같은·처럼·느낌·풍·스타일이 붙은 쪽은 비유라 버린다.
  - "music for reading books", "songs from movies": 전치사 뒤의 쪽은 수식어라 버린다.
  - 남은 것 중 마지막을 고른다. 한국어("영화 OST", "책 읽을 때 듣는 음악")와 영어 복합명사
    ("movie soundtrack")는 핵심 명사가 뒤에 온다.
  - "영화나 책", "movies or books"처럼 남은 것이 여러 도메인이고 그 사이에 '와/과/나/랑/
    and/or/쉼표'가 있으면 둘 다 원하는 것이므로 all.
"""
import re

ALL = "all"

# 한국어: 어절이 이 명사로 끝나면 단서다(합성어 "추리소설", "공포영화", "연주곡" 포함).
KO_NOUNS = {
    "movie": ("영화", "무비", "극장판"),
    "music": ("노래", "음악", "곡", "앨범", "음반", "플레이리스트", "플리", "브금", "ost", "bgm"),
    "book": ("책", "도서", "소설", "시집", "에세이", "수필", "단편집"),
}
# 그 명사로 끝나지만 작품 종류가 아닌 말
KO_EXCEPT = {
    "산책", "공책", "대책", "정책", "실책", "방책", "계책", "책책",
    "계곡", "협곡", "왜곡", "굴곡", "완곡", "희곡",
    "리플리",
}
# 어절 끝에 붙는 조사 — 떼고 본다("영화를", "노래로", "책이랑"). 긴 것부터.
KO_PARTICLES = ("이랑", "에서", "으로", "하고", "이나", "의", "에", "을", "를", "은", "는",
                "이", "가", "와", "과", "도", "로", "랑", "나")
KO_CONJ = ("와", "과", "랑", "이랑", "나", "이나", "하고")
KO_CONJ_WORDS = {"또는", "및", "혹은", "아니면"}
# 비유 표지 — 붙은 쪽은 수식어("영화 같은 노래", "영화같은", "책처럼")
KO_SIMILE_SUFFIX = ("같은", "같이", "처럼", "느낌", "느낌의", "풍", "풍의", "스러운")
KO_SIMILE_WORDS = {"같은", "같이", "처럼", "느낌", "느낌의", "느낌인", "풍", "풍의", "스타일", "스타일의"}

EN_NOUNS = {
    "movie": r"movies?|films?|cinema",
    "music": r"music|songs?|tracks?|albums?|playlists?|soundtracks?|ost|bgm",
    "book": r"books?|novels?|poetry collections?|essays?",
}
EN_RE = re.compile(r"\b(" + "|".join(f"(?P<{d}>{p})" for d, p in EN_NOUNS.items()) + r")\b", re.I)
EN_PREP = {"for", "from", "about", "like", "of", "in", "on", "based", "inspired", "to"}
EN_CONJ = {"and", "or", "&", ","}
EN_SIMILE_NEXT = re.compile(r"^-?(like|style|esque|ish)\b", re.I)


def _strip_particle(tok: str) -> tuple[str, str]:
    for p in KO_PARTICLES:
        if tok.endswith(p) and len(tok) > len(p):
            return tok[: -len(p)], p
    return tok, ""


def _ko_domain(stem: str) -> str | None:
    stem = stem.lower()                                # "영화 OST"
    if stem in KO_EXCEPT:
        return None
    for dom, nouns in KO_NOUNS.items():
        for n in nouns:
            if stem.endswith(n):
                return dom
    return None


def _korean_mentions(text: str) -> list[tuple[int, str, bool]]:
    """(어절 위치, 도메인, 뒤에 접속 표지가 붙었는지). 비유로 쓰인 것은 뺀다."""
    toks = text.split()
    out = []
    for i, tok in enumerate(toks):
        simile = False
        for s in KO_SIMILE_SUFFIX:
            if tok.endswith(s) and len(tok) > len(s):
                tok, simile = tok[: -len(s)], True
                break
        dom = _ko_domain(tok)
        conj = False
        if dom is None:
            stem, p = _strip_particle(tok)
            dom = _ko_domain(stem)
            conj = p in KO_CONJ
        if dom is None:
            continue
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if simile or nxt in KO_SIMILE_WORDS:
            continue
        conj = conj or nxt in KO_CONJ_WORDS
        out.append((i, dom, conj))
    return out


def _english_mentions(text: str) -> list[tuple[int, str, bool]]:
    out = []
    for m in EN_RE.finditer(text):
        dom = next(d for d in EN_NOUNS if m.group(d))
        before = text[: m.start()].lower().split()
        after = text[m.end():]
        # 바로 앞이나 두 단어 앞에 전치사 — "songs from movies", "music for reading books"
        if any(w.strip(",") in EN_PREP for w in before[-2:]):
            continue
        if EN_SIMILE_NEXT.match(after.strip()) or after.startswith("-like"):
            continue                                   # "movie-like songs"
        conj = bool(re.match(r"^\s*(,|and\b|or\b|&)", after, re.I))
        out.append((m.start(), dom, conj))
    return out


def _resolve(mentions: list[tuple[int, str, bool]]) -> str:
    if not mentions:
        return ALL
    doms = {d for _, d, _ in mentions}
    if len(doms) > 1 and any(c for _, _, c in mentions[:-1]):
        return ALL                                     # "영화나 책", "movies or books"
    return mentions[-1][1]


def detect_domain(text: str) -> str:
    """'all' | 'movie' | 'music' | 'book'. 검색어 원문(번역 전)을 넣는다."""
    text = (text or "").strip()
    if not text:
        return ALL
    ko = _korean_mentions(text)
    en = _english_mentions(text)
    if ko and en:
        # 한·영이 섞인 검색어("jazz 영화")는 드물다. 둘이 같은 도메인을 가리킬 때만 믿는다.
        a, b = _resolve(ko), _resolve(en)
        return a if a == b else ALL
    return _resolve(ko or en)
