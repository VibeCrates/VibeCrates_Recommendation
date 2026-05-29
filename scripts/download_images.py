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
import pandas as pd
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

DOMAIN_CONFIGS = {
    "movie": {
        "csv": "data/MovieGenre.csv",
        "id_col": "imdbId",
        "url_col": "Poster",
        "out_dir": "data/images/movie",
        "valid_url": lambda url: isinstance(url, str) and url.startswith("http"),
    },
    "music": {
        "csv": "data/music_features.csv",
        "id_col": "id",
        "url_col": "img",
        "out_dir": "data/images/music",
        "valid_url": lambda url: isinstance(url, str) and url.startswith("http") and url not in ("no", "nan"),
    },
    "book": {
        "csv": "data/Books_filtered.csv",
        "id_col": "ISBN",
        "url_col": "Image-URL-L",
        "out_dir": "data/images/book",
        "valid_url": lambda url: isinstance(url, str) and url.startswith("http"),
    },
}

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
})


def download_one(item_id: str, url: str, out_path: Path) -> tuple[str, bool]:
    if out_path.exists():
        return item_id, True
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
