"""
TMDB API로 movie_canonical.csv 각 영화의 메타데이터 수집.

수집 항목: release_date, running_time, director (list), actor (list, top 5)
IMDB ID → TMDB /find → /movie?append_to_response=credits (2 req/movie)
체크포인트: data/cache/tmdb_meta_cache.json (500건마다)
출력: movie_canonical.csv에 4컬럼을 덧붙여 그 자리에 덮어씀 (기존 8컬럼은
      건드리지 않음 — Title/text/query에 latin-1 오독으로 인한 mojibake를
      만들었던 과거 버그가 있었으므로 CSV는 반드시 UTF-8로 읽는다)

실행 예:
  TMDB_API_KEY="..." python3 scripts/fetch_movie_meta.py
  TMDB_API_KEY="..." python3 scripts/fetch_movie_meta.py --limit 100
"""
import argparse
import json
import os
import time

import pandas as pd
import requests
from tqdm import tqdm

CSV_PATH         = "data/canonical/movie_canonical.csv"
CACHE_PATH       = "data/cache/tmdb_meta_cache.json"
CHECKPOINT_EVERY = 500
REQUEST_DELAY    = 0.12   # ~8 req/s (2 calls/movie → ~4 movies/s, TMDB 한도 여유)
MAX_ACTORS       = 5

API_KEY = os.environ["TMDB_API_KEY"]
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json", "User-Agent": "VibeCrates/1.0"})


def _get(url: str, params: dict, retries: int = 3) -> dict | None:
    """GET with retry + rate-limit backoff."""
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def fetch_meta(imdb_id: str) -> dict:
    """
    IMDB ID → TMDB find → movie+credits.
    Returns dict with release_date, running_time, director, actor.
    Empty values on failure.
    """
    empty = {"release_date": "", "running_time": None, "director": [], "actor": []}

    # Step 1: IMDB ID → TMDB movie_id
    find_data = _get(
        f"https://api.themoviedb.org/3/find/tt{str(imdb_id).zfill(7)}",
        {"api_key": API_KEY, "external_source": "imdb_id"},
    )
    if not find_data:
        return empty

    results = find_data.get("movie_results", [])
    if not results:
        return empty

    tmdb_id = results[0]["id"]
    time.sleep(REQUEST_DELAY)

    # Step 2: movie details + credits
    detail_data = _get(
        f"https://api.themoviedb.org/3/movie/{tmdb_id}",
        {"api_key": API_KEY, "append_to_response": "credits"},
    )
    if not detail_data:
        return empty

    directors = [
        c["name"]
        for c in detail_data.get("credits", {}).get("crew", [])
        if c.get("job") == "Director"
    ]
    actors = [
        c["name"]
        for c in detail_data.get("credits", {}).get("cast", [])[:MAX_ACTORS]
    ]

    return {
        "release_date": detail_data.get("release_date", ""),
        "running_time": detail_data.get("runtime"),
        "director": directors,
        "actor": actors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="처리할 최대 행 수")
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH, low_memory=False)
    if args.limit:
        df = df.head(args.limit)

    cache: dict = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
    print(f"전체: {len(df):,}개 | 캐시: {len(cache):,}개 | 남은 처리: {len(df) - len(cache):,}개\n")

    remaining = [row for _, row in df.iterrows() if str(row["imdbId"]) not in cache]

    success = sum(1 for v in cache.values() if v.get("release_date") or v.get("director"))

    for i, row in enumerate(tqdm(remaining, desc="Fetching TMDB meta"), 1):
        imdb_id = str(row["imdbId"])
        meta = fetch_meta(imdb_id)
        cache[imdb_id] = meta

        if meta.get("release_date") or meta.get("director"):
            success += 1

        time.sleep(REQUEST_DELAY)

        if i % CHECKPOINT_EVERY == 0:
            with open(CACHE_PATH, "w") as f:
                json.dump(cache, f, ensure_ascii=False)
            tqdm.write(f"  체크포인트 저장 {i:,}/{len(remaining):,} | 성공: {success:,}")

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"\n캐시 저장 완료 → {CACHE_PATH}")

    # CSV 병합
    df["imdbId"] = df["imdbId"].astype(str)

    def _get_field(imdb_id: str, field: str):
        entry = cache.get(imdb_id, {})
        val = entry.get(field)
        if isinstance(val, list):
            return json.dumps(val, ensure_ascii=False)
        return val

    df["release_date"]  = df["imdbId"].map(lambda x: _get_field(x, "release_date"))
    df["running_time"]  = df["imdbId"].map(lambda x: _get_field(x, "running_time"))
    df["director"]      = df["imdbId"].map(lambda x: _get_field(x, "director"))
    df["actor"]         = df["imdbId"].map(lambda x: _get_field(x, "actor"))

    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    print(f"\n=== 수집 결과 ===")
    print(f"  release_date 수집: {(df['release_date'].fillna('') != '').sum():,} / {len(df):,}")
    print(f"  running_time 수집: {df['running_time'].notna().sum():,} / {len(df):,}")
    print(f"  director 수집    : {(df['director'].fillna('[]') != '[]').sum():,} / {len(df):,}")
    print(f"  actor 수집       : {(df['actor'].fillna('[]') != '[]').sum():,} / {len(df):,}")
    print(f"\n출력 저장 → {CSV_PATH}")


if __name__ == "__main__":
    main()
