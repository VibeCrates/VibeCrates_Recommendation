"""
Books.csv의 Image-URL-M 컬럼 유효성 검증.

Amazon CDN은 존재하지 않는 이미지에도 200을 반환하며 플레이스홀더를 전송한다.
- 43B  GIF (1×1 픽셀) : 완전 무효
- 2254B JPEG (160×107): "이미지 없음" 플레이스홀더
따라서 알려진 플레이스홀더 MD5와 대조해 무효 여부를 판정한다.

유효하지 않은 행 인덱스(헤더 제외, 0-based)를 JSON으로 저장한 뒤
최종 필터링된 CSV를 출력한다.

체크포인트 방식으로 동작하므로 중단 후 재실행하면 이어서 검증한다.
"""

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_CSV = DATA_DIR / "Books.csv"
OUTPUT_CSV = DATA_DIR / "Books_filtered.csv"
CHECKPOINT = DATA_DIR / "books_url_check.json"  # {row_idx: True/False}

# ── 검증 파라미터 ──────────────────────────────────────────────────────────────
MAX_WORKERS = 200       # 동시 요청 수
TIMEOUT = 5             # 초
SAVE_EVERY = 2000       # N건마다 체크포인트 저장
URL_COLUMN = "Image-URL-M"
STREAM_READ_BYTES = 3000  # 플레이스홀더 탐지에 필요한 최대 바이트

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Amazon CDN의 알려진 플레이스홀더 MD5 (콘텐츠 전체 해시)
# - 43B  GIF (1×1 px)   : ad4b0f606e0f8465bc4c4c170b37e1a3
# - 2254B JPEG (160×107): 52429d0be691b5c0823f37eb5cfeaae5
PLACEHOLDER_MD5S: set[str] = {
    "ad4b0f606e0f8465bc4c4c170b37e1a3",
    "52429d0be691b5c0823f37eb5cfeaae5",
}


def check_url(row_idx: int, url: str) -> tuple[int, bool]:
    """
    GET 스트림으로 최대 STREAM_READ_BYTES 바이트만 읽어 유효성 판정.

    판정 로직:
    1. 응답이 200/206 이 아니면 무효
    2. 수신 바이트가 STREAM_READ_BYTES 미만이면 전체를 받은 것 → MD5로 플레이스홀더 확인
    3. STREAM_READ_BYTES 이상이면 플레이스홀더보다 크므로 유효
    """
    import hashlib

    if not url or not url.startswith("http"):
        return row_idx, False
    try:
        with requests.get(
            url, timeout=TIMEOUT, headers=HEADERS,
            stream=True, allow_redirects=True
        ) as resp:
            if resp.status_code not in (200, 206):
                return row_idx, False

            buf = b""
            for part in resp.iter_content(chunk_size=512):
                buf += part
                if len(buf) >= STREAM_READ_BYTES:
                    return row_idx, True  # 플레이스홀더보다 크면 유효

            # STREAM_READ_BYTES 미만 → 전체 콘텐츠를 받은 것
            md5 = hashlib.md5(buf).hexdigest()
            return row_idx, md5 not in PLACEHOLDER_MD5S

    except Exception:
        return row_idx, False


def load_checkpoint() -> dict[int, bool]:
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    return {}


def save_checkpoint(results: dict[int, bool]) -> None:
    with open(CHECKPOINT, "w") as f:
        json.dump(results, f)


def validate(rows: list[dict]) -> dict[int, bool]:
    results = load_checkpoint()
    already_done = set(results.keys())

    pending = [
        (i, row[URL_COLUMN])
        for i, row in enumerate(rows)
        if i not in already_done
    ]

    if not pending:
        print("체크포인트에서 전체 결과 로드 완료.")
        return results

    total = len(pending)
    print(f"검증 대상: {total:,}개  (이미 완료: {len(already_done):,}개)")

    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(check_url, idx, url): idx for idx, url in pending}

        for future in as_completed(futures):
            idx, valid = future.result()
            results[idx] = valid
            done += 1

            if done % SAVE_EVERY == 0:
                save_checkpoint(results)
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (total - done) / rate
                print(
                    f"  {done:,}/{total:,}  "
                    f"유효: {sum(results.values()):,}  "
                    f"무효: {sum(1 for v in results.values() if not v):,}  "
                    f"남은 시간: {remaining/60:.1f}분"
                )

    save_checkpoint(results)
    return results


def write_filtered_csv(rows: list[dict], results: dict[int, bool], fieldnames: list[str]) -> None:
    valid_rows = [row for i, row in enumerate(rows) if results.get(i, False)]
    print(f"\n유효 행: {len(valid_rows):,} / 전체: {len(rows):,}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_rows)

    print(f"저장 완료: {OUTPUT_CSV}")


def main():
    print(f"CSV 로딩 중: {INPUT_CSV}")
    with open(INPUT_CSV, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    print(f"총 {len(rows):,}개 행 로드 완료\n")

    results = validate(rows)

    invalid_indices = sorted(i for i, v in results.items() if not v)
    print(f"\n무효 URL 행 수: {len(invalid_indices):,}")
    print(f"무효 행 인덱스 (처음 20개): {invalid_indices[:20]}")

    invalid_path = DATA_DIR / "books_invalid_indices.json"
    with open(invalid_path, "w") as f:
        json.dump(invalid_indices, f)
    print(f"무효 행 인덱스 저장: {invalid_path}")

    write_filtered_csv(rows, results, fieldnames)


if __name__ == "__main__":
    main()
