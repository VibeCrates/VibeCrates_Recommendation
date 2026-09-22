"""보유 아이템 중 "유명한" 것만 뽑는다 — 도메인마다 인기도 신호가 다르다.

- music: canonical에 Spotify `popularity`(0~100)가 이미 있다. 커버리지 89.8%.
- book: canonical(`book_canonical_v2.csv`)에는 인기도 신호가 없다 — 3소스 병합 때
  빠졌다. 다만 Goodreads 출처 행(`source == 'goodreads'`, 48,021권/110,594권,
  43.4%)은 원본 `data/raw/books_zenodo.csv`에 `numRatings`(평가 참여 수)가
  남아 있어서, `asin`의 `gr_` 접두어를 뗀 값(원본 `bookId`)으로 다시 조인해
  복구할 수 있다. BX(42,823권)·Kindle(19,750권) 출처는 원본에도 이 신호가 없어
  이번엔 뺀다.
- movie: canonical의 `IMDB Score`는 평점이지 인기도가 아니었다(평가 참여 수가
  없어 소수 평가로 받은 고득점과 구분이 안 됨). 2026-09-18~19에 TMDB API로
  `vote_count`·`popularity`를 추가 수집해 해소했다(`scripts/fetch_movie_meta.py`,
  커버리지 98.3%) — `vote_count`(평가 참여 수)로 뽑는다.

사용:
  python scripts/extract_popular_items.py --top 200
"""
import argparse

import pandas as pd


def extract_movie(top: int) -> pd.DataFrame:
    df = pd.read_csv("data/canonical/movie_canonical.csv", low_memory=False)
    df = df[df["vote_count"].notna()].copy()
    df = df.sort_values("vote_count", ascending=False)
    cols = ["imdbId", "Title", "Genre", "vote_count", "popularity", "IMDB Score"]
    return df[cols].head(top)


def extract_music(top: int) -> pd.DataFrame:
    df = pd.read_csv("data/canonical/music_canonical.csv")
    df = df[df["popularity"].notna()].copy()
    df = df.sort_values("popularity", ascending=False)
    cols = ["id", "name", "artists", "genre", "popularity",
            "total_artist_followers", "avg_artist_popularity"]
    return df[cols].head(top)


def extract_book_goodreads(top: int) -> pd.DataFrame:
    b = pd.read_csv("data/canonical/book_canonical_v2.csv", low_memory=False)
    gr = b[b["source"] == "goodreads"].copy()
    gr["bookId"] = gr["asin"].str.replace("^gr_", "", regex=True)

    z = pd.read_csv("data/raw/books_zenodo.csv", usecols=["bookId", "numRatings", "rating"])
    z = z.drop_duplicates("bookId")  # 원본에 완전 중복 행 54건 존재(값도 동일)

    merged = gr.merge(z, on="bookId", how="left")
    unmatched = merged["numRatings"].isna().sum()
    if unmatched:
        print(f"[경고] goodreads {len(gr):,}권 중 {unmatched:,}권이 원본과 매칭 안 됨")

    merged = merged[merged["numRatings"].notna()].sort_values("numRatings", ascending=False)
    cols = ["asin", "title", "author", "category_name", "numRatings", "rating"]
    return merged[cols].head(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--out-prefix", default="experiments/popular")
    args = ap.parse_args()

    movie = extract_movie(args.top)
    movie.to_csv(f"{args.out_prefix}_movie.csv", index=False)
    print(f"movie: vote_count 상위 {len(movie)}편 → {args.out_prefix}_movie.csv")
    print(movie.head(10)[["Title", "vote_count", "popularity"]].to_string(index=False))

    print()
    music = extract_music(args.top)
    music.to_csv(f"{args.out_prefix}_music.csv", index=False)
    print(f"music: popularity 상위 {len(music)}곡 → {args.out_prefix}_music.csv")
    print(music.head(10)[["name", "artists", "popularity"]].to_string(index=False))

    print()
    book = extract_book_goodreads(args.top)
    book.to_csv(f"{args.out_prefix}_book_goodreads.csv", index=False)
    print(f"book(goodreads만): numRatings 상위 {len(book)}권 → {args.out_prefix}_book_goodreads.csv")
    print(book.head(10)[["title", "author", "numRatings", "rating"]].to_string(index=False))

    # id + 이름만 남긴 요약본. 세 도메인 합쳐 한 파일로 - id는 도메인마다 원래
    # id 컬럼 그대로다(movie=imdbId, music=id, book=asin).
    ids = pd.concat([
        movie.rename(columns={"imdbId": "id", "Title": "name"})[["id", "name"]].assign(domain="movie"),
        music.rename(columns={"name": "name"})[["id", "name"]].assign(domain="music"),
        book.rename(columns={"asin": "id", "title": "name"})[["id", "name"]].assign(domain="book"),
    ], ignore_index=True)[["domain", "id", "name"]]
    ids.to_csv(f"{args.out_prefix}_top{args.top}_ids.csv", index=False)
    print(f"\nid+이름 요약본({len(ids)}행) → {args.out_prefix}_top{args.top}_ids.csv")


if __name__ == "__main__":
    main()
