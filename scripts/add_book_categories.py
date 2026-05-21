"""
Books_filtered.csv에 Open Library API로 category 컬럼을 추가.

- ISBN 100개씩 배치 요청 → 전체 ~1,500 요청
- ThreadPoolExecutor로 병렬 처리
- 2,000건마다 체크포인트 저장 (중단 후 재실행 가능)
- subjects 중 첫 번째 값을 category로 사용, 없으면 빈 문자열
"""

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_CSV = DATA_DIR / "Books_filtered.csv"
OUTPUT_CSV = DATA_DIR / "Books_filtered.csv"
CHECKPOINT = DATA_DIR / "books_category_cache.json"  # {ISBN: category}

# ── 파라미터 ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 100
MAX_WORKERS = 20
TIMEOUT = 15
SAVE_EVERY = 2000  # 누적 처리 건수 기준


def fetch_batch(isbns: list[str]) -> dict[str, str]:
    """ISBN 배치 → {ISBN: category} 반환. API 실패 시 빈 딕셔너리."""
    keys = ",".join(f"ISBN:{i}" for i in isbns)
    try:
        r = requests.get(
            f"https://openlibrary.org/api/books?bibkeys={keys}&format=json&jscmd=data",
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}

    result = {}
    for key, info in data.items():
        isbn = key.replace("ISBN:", "")
        subjects = info.get("subjects", [])
        result[isbn] = subjects[0]["name"] if subjects else ""
    return result


def load_checkpoint() -> dict[str, str]:
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {}


def save_checkpoint(cache: dict[str, str]) -> None:
    with open(CHECKPOINT, "w") as f:
        json.dump(cache, f)


def main():
    print(f"CSV 로딩 중: {INPUT_CSV}")
    with open(INPUT_CSV, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    print(f"총 {len(rows):,}개 행 로드 완료")

    cache = load_checkpoint()
    print(f"캐시 로드: {len(cache):,}개\n")

    # 미조회 ISBN 배치 구성
    pending_isbns = [r["ISBN"] for r in rows if r["ISBN"] not in cache]
    batches = [
        pending_isbns[i : i + BATCH_SIZE]
        for i in range(0, len(pending_isbns), BATCH_SIZE)
    ]
    print(f"조회 대상: {len(pending_isbns):,}개 → {len(batches):,}배치")

    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_batch, batch): batch for batch in batches}

        for future in as_completed(futures):
            cache.update(future.result())
            done += len(futures[future])

            if done % SAVE_EVERY < BATCH_SIZE:
                save_checkpoint(cache)
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (len(pending_isbns) - done) / rate if rate else 0
                filled = sum(1 for v in cache.values() if v)
                print(
                    f"  {done:,}/{len(pending_isbns):,}  "
                    f"카테고리 있음: {filled:,}  "
                    f"남은 시간: {remaining/60:.1f}분"
                )

    save_checkpoint(cache)

    # category 컬럼 추가 및 저장
    if "category" not in fieldnames:
        fieldnames.append("category")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["category"] = cache.get(row["ISBN"], "")
            writer.writerow(row)

    filled = sum(1 for r in rows if cache.get(r["ISBN"]))
    print(f"\n카테고리 채워진 행: {filled:,} / {len(rows):,} ({filled/len(rows)*100:.1f}%)")
    print(f"저장 완료: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
