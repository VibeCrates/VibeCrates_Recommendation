"""
DB 스키마와의 자료형 불일치를 최소화하기 위한 변환 스크립트.

변환 항목:
  [Movie]  Genre       : "A|B|C"          → JSON array string ["A","B","C"]
           Title       : "Toy Story (1995)"→ "Toy Story" (연도 제거)
           director    : already JSON str  → 그대로 유지
           actor       : already JSON str  → 그대로 유지
           running_time: float → int (null 유지)

  [Music]  genre       : "Hip-Hop"         → JSON array string ["Hip-Hop"]
           year        : 2020.0 (float)    → "2020-01-01" (date string, NaN → "")

  [Book]   category_name: "Parenting ..."  → JSON array string ["Parenting ..."]
           publishedDate: already YYYY-MM-DD → 그대로 유지

DB 적재용 산출물은 data/db_export/ 아래 별도 파일로 생성하며,
학습 파이프라인이 읽는 원본 CSV(정본)는 절대 덮어쓰지 않는다.
(원본을 in-place로 덮어쓰면 genre/category_name이 JSON array string으로
 바뀌어 preprocessing.py의 _build_content_text가 만드는 학습 텍스트가
 손상되는 문제가 있었음 — session 14 참조)

실행:
  python3 scripts/transform_datasets.py
  python3 scripts/transform_datasets.py --skip-movie   # Movie는 TMDB 수집 완료 후 실행
"""
import argparse
import json
import os
import re

import pandas as pd

DB_EXPORT_DIR = "data/db_export"

MOVIE_INPUT  = "data/canonical/movie_canonical.csv"
MOVIE_OUTPUT = os.path.join(DB_EXPORT_DIR, "MovieGenre_db.csv")

MUSIC_INPUT  = "data/canonical/music_canonical.csv"
MUSIC_OUTPUT = os.path.join(DB_EXPORT_DIR, "music_features_db.csv")

BOOK_INPUT   = "data/canonical/book_canonical.csv"
BOOK_OUTPUT  = os.path.join(DB_EXPORT_DIR, "kindle_data_db.csv")


def to_json_array(val, sep: str | None = None) -> str:
    """단일 string 또는 구분자 기반 string을 JSON array string으로 변환."""
    if pd.isna(val) or str(val).strip() == "":
        return "[]"
    s = str(val).strip()
    if sep:
        items = [x.strip() for x in s.split(sep) if x.strip()]
    else:
        items = [s]
    return json.dumps(items, ensure_ascii=False)


def transform_movie(df: pd.DataFrame) -> pd.DataFrame:
    print("[Movie] 변환 시작...")

    # Title: "Toy Story (1995)" → "Toy Story"
    df["Title"] = df["Title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()

    # Genre: "Animation|Adventure|Comedy" → ["Animation","Adventure","Comedy"]
    df["Genre"] = df["Genre"].apply(lambda v: to_json_array(v, sep="|"))

    # running_time: float → int (결측은 pd.NA 유지)
    df["running_time"] = pd.to_numeric(df["running_time"], errors="coerce").astype("Int64")

    # director / actor: 이미 JSON string이므로 유효성만 확인 후 유지
    for col in ("director", "actor"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: v if pd.notna(v) else "[]")

    print(f"  Title 변환 완료 (샘플: {df['Title'].iloc[0]!r})")
    print(f"  Genre 변환 완료 (샘플: {df['Genre'].iloc[0]})")
    print(f"  running_time 타입: {df['running_time'].dtype}")
    return df


def transform_music(df: pd.DataFrame) -> pd.DataFrame:
    print("[Music] 변환 시작...")

    # genre: "Hip-Hop" → ["Hip-Hop"]
    df["genre"] = df["genre"].apply(lambda v: to_json_array(v))

    # year: 2020.0 → "2020-01-01", NaN → ""
    def year_to_date(v):
        try:
            return f"{int(v)}-01-01"
        except (ValueError, TypeError):
            return ""
    df["year"] = df["year"].apply(year_to_date)

    print(f"  genre 변환 완료 (샘플: {df['genre'].iloc[0]})")
    print(f"  year 변환 완료  (샘플: {df['year'].iloc[0]!r})")
    return df


def transform_book(df: pd.DataFrame) -> pd.DataFrame:
    print("[Book] 변환 시작...")

    # category_name: "Parenting & Relationships" → ["Parenting & Relationships"]
    df["category_name"] = df["category_name"].apply(lambda v: to_json_array(v))

    print(f"  category_name 변환 완료 (샘플: {df['category_name'].iloc[0]})")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-movie", action="store_true", help="Movie 변환 건너뜀 (TMDB 수집 미완료 시)")
    args = parser.parse_args()

    os.makedirs(DB_EXPORT_DIR, exist_ok=True)

    # Movie
    if args.skip_movie:
        print("[Movie] 건너뜀 (--skip-movie)\n")
    elif not os.path.exists(MOVIE_INPUT):
        print(f"[Movie] {MOVIE_INPUT} 없음 — TMDB 수집 완료 후 재실행 필요\n")
    else:
        df = pd.read_csv(MOVIE_INPUT, low_memory=False)
        df = transform_movie(df)
        df.to_csv(MOVIE_OUTPUT, index=False, encoding="utf-8")
        print(f"  저장 → {MOVIE_OUTPUT}\n")

    # Music
    df = pd.read_csv(MUSIC_INPUT, low_memory=False)
    df = transform_music(df)
    df.to_csv(MUSIC_OUTPUT, index=False, encoding="utf-8")
    print(f"  저장 → {MUSIC_OUTPUT}\n")

    # Book
    df = pd.read_csv(BOOK_INPUT, low_memory=False)
    df = transform_book(df)
    df.to_csv(BOOK_OUTPUT, index=False, encoding="utf-8")
    print(f"  저장 → {BOOK_OUTPUT}\n")

    print("=== 완료 ===")
    print("Movie 변환은 TMDB 수집 완료 후 `python3 scripts/transform_datasets.py` 재실행")


if __name__ == "__main__":
    main()
