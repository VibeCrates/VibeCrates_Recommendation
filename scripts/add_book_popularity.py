"""
Books_filtered.csv에 Open Library Search API로 popularity 관련 컬럼 추가.

수집 지표:
  - edition_count      : 출판 판본 수 (역사적 인기 / 고전 강도)
  - already_read_count : OL 유저 "읽음" 표시 수
  - want_to_read_count : OL 유저 "읽고 싶음" 표시 수
  - ratings_average    : 평균 평점 (없으면 None)
  - ratings_count      : 평점 참여 수

popularity_score (파생 컬럼):
  already_read_count + want_to_read_count + edition_count * 10

ISBN 1개씩 개별 요청 (배치 불가).
ThreadPoolExecutor로 병렬 처리, 체크포인트 저장.
"""

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ── 경로 ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_CSV = DATA_DIR / "Books_filtered.csv"
OUTPUT_CSV = DATA_DIR / "Books_filtered.csv"
CHECKPOINT = DATA_DIR / "books_popularity_cache.json"  # {ISBN: {...}}

# ── 파라미터 ──────────────────────────────────────────────────────────────────
MAX_WORKERS = 50
TIMEOUT = 10
SAVE_EVERY = 2000

FIELDS = "edition_count,ratings_average,ratings_count,want_to_read_count,currently_reading_count,already_read_count"

NEW_COLUMNS = [
    "edition_count",
    "already_read_count",
    "want_to_read_count",
    "ratings_average",
    "ratings_count",
    "popularity_score",
]


def fetch_popularity(isbn: str) -> dict:
    """Open Library Search API로 인기도 지표 수집."""
    try:
        r = requests.get(
            f"https://openlibrary.org/search.json?isbn={isbn}&fields={FIELDS}",
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        docs = r.json().get("docs", [])
        if not docs:
            return {}
        d = docs[0]
        return {
            "edition_count":      d.get("edition_count") or 0,
            "already_read_count": d.get("already_read_count") or 0,
            "want_to_read_count": d.get("want_to_read_count") or 0,
            "ratings_average":    d.get("ratings_average"),
            "ratings_count":      d.get("ratings_count") or 0,
        }
    except Exception:
        return {}


def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {}


def save_checkpoint(cache: dict) -> None:
    with open(CHECKPOINT, "w") as f:
        json.dump(cache, f)


def main():
    print(f"CSV 로딩: {INPUT_CSV}")
    with open(INPUT_CSV, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    print(f"총 {len(rows):,}개 행\n")

    cache = load_checkpoint()
    print(f"캐시 로드: {len(cache):,}개")

    pending = [r["ISBN"] for r in rows if r["ISBN"] not in cache]
    print(f"조회 대상: {len(pending):,}개\n")

    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_popularity, isbn): isbn for isbn in pending}

        for future in as_completed(futures):
            isbn = futures[future]
            cache[isbn] = future.result()
            done += 1

            if done % SAVE_EVERY == 0:
                save_checkpoint(cache)
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (len(pending) - done) / rate if rate else 0
                print(
                    f"  {done:,}/{len(pending):,}  "
                    f"남은 시간: {remaining/60:.1f}분"
                )

    save_checkpoint(cache)
    print("수집 완료. CSV 저장 중...")

    for col in NEW_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)

    for row in rows:
        info = cache.get(row["ISBN"], {})
        edition      = info.get("edition_count", 0) or 0
        already_read = info.get("already_read_count", 0) or 0
        want_to_read = info.get("want_to_read_count", 0) or 0

        row["edition_count"]      = edition
        row["already_read_count"] = already_read
        row["want_to_read_count"] = want_to_read
        row["ratings_average"]    = info.get("ratings_average", "")
        row["ratings_count"]      = info.get("ratings_count", 0) or 0
        row["popularity_score"]   = already_read + want_to_read + edition * 10

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    scores = [int(r["popularity_score"]) for r in rows]
    scores.sort(reverse=True)
    print(f"\n저장 완료: {OUTPUT_CSV}")
    print(f"popularity_score 상위 5: {scores[:5]}")
    print(f"popularity_score 중앙값: {scores[len(scores)//2]}")
    print(f"popularity_score > 0 인 행: {sum(1 for s in scores if s > 0):,}개")


if __name__ == "__main__":
    main()
