"""
세 개의 도서 소스를 하나의 canonical CSV로 병합한다 (블러브 온전 행만).

동기 (docs/TODO.md 0단계):
  기존 Kindle canonical은 133,102권이지만 블러브(=description) 보유가 15.0%뿐이라,
  85%는 content_text가 제목/저자/카테고리로만 채워져 학습 신호가 사실상 없다.
  세 소스는 모집단이 거의 겹치지 않으므로(Kindle 롱테일 / Goodreads 유명작 큐레이션 /
  BX 2004년 종이책 카탈로그), "빈 블러브를 채우는" 전략으로는 8%밖에 못 메운다.
  대신 각 소스에서 **블러브가 온전한 행만** 취해 합집합을 만든다.

우선순위 (중복 시 앞선 소스를 남김):
  Kindle > Goodreads > BX
  — Kindle은 쿼리·이미지·카테고리를 모두 갖고 있고, Goodreads는 커버·장르가 있으며,
    BX는 이미지를 크롤링해야 하고 장르가 없다.

출력 스키마는 기존 book_canonical.csv와 호환된다 (DOMAIN_CONFIG의 id_col="asin",
image_col="imgUrl"을 그대로 쓰므로 파이프라인 수정 없이 교체 가능).

사용:
  python scripts/merge_book_sources.py
  python scripts/merge_book_sources.py --output data/canonical/book_canonical_v2.csv
"""

import re
import ast
import argparse

import pandas as pd

from scripts.clean_book_description import clean_description

KINDLE_CSV = "data/canonical/book_canonical.csv"
GOODREADS_CSV = "data/raw/books_zenodo.csv"      # Zenodo 원본 (Kaggle 재업로드본은 ISBN 파괴됨)
BX_CSV = "data/raw/books_with_blurbs.csv"
DEFAULT_OUT = "data/canonical/book_canonical_v2.csv"

# Open Library 커버 API. default=false가 없으면 커버 없는 책에 43바이트 빈 이미지가
# 200으로 돌아와 조용히 학습 데이터를 오염시킨다 (docs/TODO.md 1단계 참조).
OL_COVER = "https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg?default=false"

ISBN_PLACEHOLDER = "9999999999999"   # Goodreads 원본이 결측 ISBN에 쓰는 값

OUT_COLUMNS = [
    "asin", "title", "author", "category_name",
    "description", "description_clean", "imgUrl",
    "publishedDate", "query", "isbn", "source",
]


# ── 정규화 유틸 ───────────────────────────────────────────────────────────────

def blank(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype(str).str.strip().isin(["", "nan", "None"])


def norm_title(s) -> str:
    """중복 판정용 제목 키. 괄호(시리즈 표기)와 콜론 뒤 부제를 떼어낸다."""
    s = str(s).lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = s.split(":")[0]
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_author(s) -> str:
    """'Rowling, J.K.' 와 'J.K. Rowling'을 같은 키로 만든다 (토큰 집합 정렬)."""
    s = str(s).lower().split("(")[0]
    toks = {t for t in re.split(r"[^a-z]+", s) if len(t) > 1}
    return " ".join(sorted(toks))


def dedup_key(df: pd.DataFrame) -> pd.Series:
    return df["title"].map(norm_title) + "|" + df["author"].map(norm_author)


# Goodreads 저자명에는 역할 주석이 붙어 있다 ("J.K. Rowling, Mary GrandPré (Illustrator)",
# "(Goodreads Author)" 27,465행). 그대로 두면 content_text에 메타 노이즈로 실린다.
AUTHOR_ROLE = re.compile(
    r"\s*\((goodreads author|illustrator|translator|editor|foreword|introduction|"
    r"contributor|narrator|photographer|adapter|afterword|preface)[^)]*\)",
    re.I,
)


def strip_author_roles(v) -> str:
    if pd.isna(v):
        return ""
    s = AUTHOR_ROLE.sub("", str(v))
    s = re.sub(r"\s*,\s*", ", ", s).strip(" ,")
    return re.sub(r"\s+", " ", s)


# 스크랩 아티팩트: 문장 끝 뒤에 붙은 미아 쉼표("...disease. ,In 1918") 13,216행.
# 원문 문단 구분이 쉼표로 치환되면서 생긴 것으로 보인다. 보수적으로 두 패턴만 정리한다.
STRAY_COMMA = re.compile(r"([.!?])\s*,\s*")
SPACE_COMMA = re.compile(r"\s+,")


def normalize_punct(v) -> str:
    if pd.isna(v):
        return ""
    s = STRAY_COMMA.sub(r"\1 ", str(v))
    s = SPACE_COMMA.sub(",", s)
    return re.sub(r"[ \t]+", " ", s).strip()


def valid_isbn(v) -> str:
    """ISBN-10/13 형식만 통과. 플레이스홀더와 엑셀 지수표기는 버린다."""
    s = str(v).strip().upper()
    if s in ("", "NAN", "NONE", ISBN_PLACEHOLDER):
        return ""
    return s if re.fullmatch(r"\d{9}[\dX]|\d{13}", s) else ""


def flatten_genres(v, limit: int = 5) -> str:
    """"['Young Adult', 'Fiction', ...]" → "Young Adult, Fiction, ..." """
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if not s or s == "[]":
        return ""
    try:
        items = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s
    if not isinstance(items, list):
        return s
    return ", ".join(str(i).strip() for i in items[:limit] if str(i).strip())


def goodreads_year(v) -> str:
    """Goodreads publishDate는 '09/14/08' 형식. 2자리 연도를 4자리로 편다."""
    if pd.isna(v):
        return ""
    ts = pd.to_datetime(str(v).strip(), format="%m/%d/%y", errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(str(v).strip(), errors="coerce")
    return "" if pd.isna(ts) else str(ts.year)


def bx_year(v) -> str:
    try:
        y = int(float(v))
    except (TypeError, ValueError):
        return ""
    return str(y) if 1400 <= y <= 2100 else ""


# ── 소스별 로더 ───────────────────────────────────────────────────────────────

def load_kindle() -> pd.DataFrame:
    df = pd.read_csv(KINDLE_CSV, low_memory=False)
    keep = df[~blank(df["description_clean"])].copy()
    out = pd.DataFrame({
        "asin": "kdl_" + keep["asin"].astype(str),
        "title": keep["title"].astype(str),
        "author": keep["author"].fillna("").astype(str),
        "category_name": keep["category_name"].fillna("").astype(str),
        "description": keep["description"].fillna("").astype(str),
        "description_clean": keep["description_clean"].astype(str).map(normalize_punct),
        "imgUrl": keep["imgUrl"].fillna("").astype(str),
        "publishedDate": keep["publishedDate"].fillna("").astype(str),
        # 기존 쿼리를 그대로 승계 — 재생성 비용을 아끼는 유일한 소스
        "query": keep["query"].fillna("").astype(str),
        "isbn": "",
        "source": "kindle",
    })
    return out


def load_goodreads() -> pd.DataFrame:
    df = pd.read_csv(GOODREADS_CSV, low_memory=False)
    keep = df[~blank(df["description"])].copy()
    desc = keep["description"].astype(str)
    out = pd.DataFrame({
        "asin": "gr_" + keep["bookId"].astype(str),
        "title": keep["title"].astype(str),
        "author": keep["author"].map(strip_author_roles),
        "category_name": keep["genres"].map(flatten_genres),
        "description": desc,
        "description_clean": desc.map(clean_description).map(normalize_punct),
        "imgUrl": keep["coverImg"].fillna("").astype(str),
        "publishedDate": keep["publishDate"].map(goodreads_year),
        "query": "",
        "isbn": keep["isbn"].map(valid_isbn),
        "source": "goodreads",
    })
    return out


def load_bx() -> pd.DataFrame:
    df = pd.read_csv(BX_CSV, low_memory=False)
    keep = df[~blank(df["Blurb"])].copy()
    isbn = keep["ISBN"].map(valid_isbn)
    desc = keep["Blurb"].astype(str)
    out = pd.DataFrame({
        "asin": "bx_" + keep["ISBN"].astype(str).str.strip(),
        "title": keep["Title"].astype(str),
        "author": keep["Author"].fillna("").astype(str),
        # BX에는 장르 컬럼이 없다 (docs/TODO.md 결정대기 항목)
        "category_name": "",
        "description": desc,
        "description_clean": desc.map(clean_description).map(normalize_punct),
        # 이미지가 없으므로 ISBN으로 Open Library URL을 미리 조립해둔다.
        # 실제 확보는 1단계 크롤링에서 (히트율 실측 94.0%).
        "imgUrl": isbn.map(lambda s: OL_COVER.format(isbn=s) if s else ""),
        "publishedDate": keep["Year"].map(bx_year),
        "query": "",
        "isbn": isbn,
        "source": "bx",
    })
    return out


# ── 병합 ──────────────────────────────────────────────────────────────────────

def merge(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """소스 내 중복 제거 후, 앞선 소스를 우선해 소스 간 중복 제거."""
    merged = []
    seen: set[str] = set()

    for df in frames:
        src = df["source"].iloc[0] if len(df) else "?"
        before = len(df)

        df = df.copy()
        df["_k"] = dedup_key(df)
        # 정제 후 블러브가 비어버린 행 제거 (마케팅 문구만으로 이루어진 경우)
        df = df[~blank(df["description_clean"])]
        after_clean = len(df)

        df = df.drop_duplicates(subset="asin", keep="first")
        df = df.drop_duplicates(subset="_k", keep="first")
        after_self = len(df)

        df = df[~df["_k"].isin(seen)]
        after_cross = len(df)

        seen.update(df["_k"])
        merged.append(df)
        print(f"  {src:10s} {before:>7,} → 정제후 {after_clean:>7,} "
              f"→ 자체중복제거 {after_self:>7,} → 소스간중복제거 {after_cross:>7,}")

    out = pd.concat(merged, ignore_index=True).drop(columns="_k")
    return out[OUT_COLUMNS]


def report(df: pd.DataFrame) -> None:
    print(f"\n=== 최종 {len(df):,}권 ===")
    print("\n소스 구성:")
    for src, n in df["source"].value_counts().items():
        print(f"  {src:10s} {n:>7,} ({n / len(df) * 100:5.1f}%)")

    print("\n컬럼 커버리지:")
    for c in ["description_clean", "imgUrl", "category_name", "query", "isbn", "publishedDate"]:
        ok = (~blank(df[c])).sum()
        note = ""
        if c == "imgUrl":
            # BX의 URL은 아직 조립만 된 상태(Open Library 히트율 94.0%)라 실제 확보량과 다르다.
            bx_url = ((df["source"] == "bx") & ~blank(df["imgUrl"])).sum()
            expect = ok - bx_url + int(bx_url * 0.94)
            note = f"  ← 크롤링 후 실제 확보 예상 {expect:,} ({expect / len(df) * 100:.1f}%)"
        print(f"  {c:18s} {ok:>7,} ({ok / len(df) * 100:5.1f}%){note}")

    print("\n소스별 이미지/쿼리/장르 보유:")
    for src, sub in df.groupby("source"):
        img = (~blank(sub["imgUrl"])).sum()
        q = (~blank(sub["query"])).sum()
        cat = (~blank(sub["category_name"])).sum()
        print(f"  {src:10s} 이미지 {img:>6,} | 쿼리 {q:>6,} | 장르 {cat:>6,}  (n={len(sub):,})")

    # 다음 단계 작업량
    need_q = blank(df["query"]).sum()
    need_crawl = (df["source"] == "bx").sum()
    print(f"\n다음 단계 작업량:")
    print(f"  Open Library 크롤링 대상 : {need_crawl:>7,} (히트율 94.0% → 약 {int(need_crawl * .94):,}장 예상)")
    print(f"  쿼리 생성 필요           : {need_q:>7,}")
    print(f"  description_synth 합성   : {len(df):>7,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=DEFAULT_OUT)
    args = ap.parse_args()

    print("소스 로딩 (블러브 온전 행만):")
    frames = [load_kindle(), load_goodreads(), load_bx()]
    print("\n중복 제거 (우선순위 Kindle > Goodreads > BX):")
    df = merge(frames)
    report(df)

    df.to_csv(args.output, index=False)
    print(f"\n저장 → {args.output}")


if __name__ == "__main__":
    main()
