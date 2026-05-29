"""
TMDB API로 MovieGenre.csv 각 영화의 text 컬럼 수집.

text = overview + tagline + keywords 조합 (SBERT 입력용)
IMDB ID → TMDB /find → /movie?append_to_response=keywords (1 req/movie)
체크포인트: data/movie_text_cache.json (500건마다)

실행 예:
  python3 scripts/fetch_movie_descriptions.py
  python3 scripts/fetch_movie_descriptions.py --limit 100
"""
import argparse
import json
import os
import time

import pandas as pd
import requests
from tqdm import tqdm

CSV_PATH         = "data/MovieGenre.csv"
CACHE_PATH       = "data/movie_text_cache.json"
CHECKPOINT_EVERY = 500
REQUEST_DELAY    = 0.05   # TMDB 50 req/s 제한 여유분

API_KEY = "4151e3ad80ae9a608e0b71bcfa4b6f19"
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json", "User-Agent": "VibeCrates/1.0"})


def fetch_text(imdb_id: str) -> str:
    """
    IMDB ID → TMDB movie_id → overview + tagline + keywords 조합 텍스트.
    실패 시 빈 문자열.
    """
    try:
        # 1) IMDB ID로 TMDB movie_id 조회
        r = SESSION.get(
            f"https://api.themoviedb.org/3/find/tt{str(imdb_id).zfill(7)}",
            params={"api_key": API_KEY, "external_source": "imdb_id"},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("movie_results", [])
        if not results:
            return ""
        tmdb_id = results[0]["id"]
        time.sleep(REQUEST_DELAY)

        # 2) overview + tagline + keywords 한 번에 조회
        r2 = SESSION.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": API_KEY, "append_to_response": "keywords"},
            timeout=10,
        )
        r2.raise_for_status()
        data = r2.json()

        parts = []
        overview = data.get("overview", "").strip()
        if overview:
            parts.append(overview)

        tagline = data.get("tagline", "").strip()
        if tagline:
            parts.append(tagline)

        keywords = [k["name"] for k in data.get("keywords", {}).get("keywords", [])]
        if keywords:
            parts.append(", ".join(keywords))

        return " | ".join(parts)

    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH, low_memory=False, encoding="latin-1")
    if args.limit:
        df = df.head(args.limit)

    cache: dict = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}

    remaining = [row for _, row in df.iterrows() if str(row["imdbId"]) not in cache]
    print(f"전체: {len(df):,}개 | 캐시: {len(cache):,}개 | 남은 처리: {len(remaining):,}개\n")

    success = sum(1 for v in cache.values() if v)

    for i, row in enumerate(tqdm(remaining, desc="Fetching"), 1):
        imdb_id = str(row["imdbId"])
        text = fetch_text(imdb_id)
        cache[imdb_id] = text
        if text:
            success += 1

        time.sleep(REQUEST_DELAY)

        if i % CHECKPOINT_EVERY == 0:
            with open(CACHE_PATH, "w") as f:
                json.dump(cache, f, ensure_ascii=False)
            print(f"  {i:,}/{len(remaining):,} | 수집 성공: {success:,}")

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, ensure_ascii=False)

    df["imdbId"] = df["imdbId"].astype(str)
    df["text"] = df["imdbId"].map(cache).fillna("")
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    filled = (df["text"].str.strip() != "").sum()
    print(f"\n완료! text 수집: {filled:,} / {len(df):,}")
    print(f"저장: {CSV_PATH}")


if __name__ == "__main__":
    main()
