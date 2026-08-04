"""
Movie/Music/Book 이미지를 로컬에 일괄 다운로드.
URL이 없거나 다운로드 실패한 항목은 건너뜁니다.

저장 경로:
  data/images/movie/{imdbId}.jpg
  data/images/music/{id}.jpg
  data/images/book/{ISBN}.jpg

실행 예:
  python3 scripts/download_images.py --domain movie
  python3 scripts/download_images.py --domain all
  python3 scripts/download_images.py --domain music --workers 20
"""

import os
import argparse
import threading
import pandas as pd
import requests
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

DOMAIN_CONFIGS = {
    "movie": {
        "csv": "data/canonical/movie_canonical.csv",
        "id_col": "imdbId",
        "url_col": "Poster",
        "out_dir": "data/images/movie",
        "valid_url": lambda url: isinstance(url, str) and url.startswith("http"),
    },
    "music": {
        "csv": "data/canonical/music_canonical.csv",
        "id_col": "id",
        "url_col": "img",
        "out_dir": "data/images/music",
        "valid_url": lambda url: isinstance(url, str) and url.startswith("http") and url not in ("no", "nan"),
    },
    "book": {
        # v2 = 3소스 병합본. BX 42,823권의 imgUrl은 Open Library 커버 API URL이며
        # merge_book_sources.py에서 `?default=false`를 붙여 조립해뒀다 —
        # 이게 없으면 커버 없는 책에 43바이트 빈 이미지가 200으로 돌아와
        # 해당 책들이 전부 동일한 z_image를 갖게 된다. default=false면 404가 나서
        # raise_for_status가 걸러낸다 (아래 1KB 하한이 2차 방어선).
        "csv": "data/canonical/book_canonical_v2.csv",
        "id_col": "asin",
        "url_col": "imgUrl",
        "out_dir": "data/images/book",
        "valid_url": lambda url: isinstance(url, str) and url.startswith("http"),
    },
}

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
})

# 호스트별 동시 요청 상한. book 도메인은 Amazon CDN / Goodreads CDN / Open Library가
# 섞여 있는데, 앞의 둘은 30~50 req/s를 받아내지만 Open Library는 실측 4.8 req/s
# (worker 6, 레이트리밋 에러 0)가 한계다. --workers를 그대로 태우면 OL만 막힌다.
HOST_LIMITS = {"covers.openlibrary.org": 6}
_host_locks: dict[str, threading.Semaphore] = {}
_locks_guard = threading.Lock()


def host_slot(url: str) -> threading.Semaphore | None:
    host = urlparse(url).netloc
    limit = HOST_LIMITS.get(host)
    if limit is None:
        return None
    with _locks_guard:
        if host not in _host_locks:
            _host_locks[host] = threading.Semaphore(limit)
        return _host_locks[host]


def download_one(item_id: str, url: str, out_path: Path) -> tuple[str, bool]:
    if out_path.exists():
        return item_id, True
    slot = host_slot(url)
    if slot is not None:
        slot.acquire()
    try:
        r = SESSION.get(url, timeout=15, stream=True)
        r.raise_for_status()
        content = r.content
        # 1KB 미만이면 플레이스홀더로 간주
        if len(content) < 1024:
            return item_id, False
        out_path.write_bytes(content)
        return item_id, True
    except Exception:
        return item_id, False
    finally:
        if slot is not None:
            slot.release()


def run_domain(domain: str, workers: int):
    cfg = DOMAIN_CONFIGS[domain]
    df = pd.read_csv(cfg["csv"], low_memory=False)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for _, row in df.iterrows():
        url = str(row.get(cfg["url_col"], ""))
        if not cfg["valid_url"](url):
            continue
        item_id = str(row[cfg["id_col"]])
        ext = ".jpg"
        out_path = out_dir / f"{item_id}{ext}"
        tasks.append((item_id, url, out_path))

    print(f"[{domain}] 다운로드 대상: {len(tasks):,}개 / 전체 {len(df):,}개")

    success = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, *t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=domain):
            _, ok = future.result()
            if ok:
                success += 1
            else:
                fail += 1

    print(f"[{domain}] 완료 — 성공: {success:,} | 실패: {fail:,}")
    print(f"[{domain}] 저장 경로: {out_dir}/\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="all", choices=["movie", "music", "book", "all"])
    parser.add_argument("--workers", type=int, default=30, help="동시 다운로드 스레드 수")
    args = parser.parse_args()

    domains = list(DOMAIN_CONFIGS.keys()) if args.domain == "all" else [args.domain]
    for domain in domains:
        run_domain(domain, args.workers)


if __name__ == "__main__":
    main()
