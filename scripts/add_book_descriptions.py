"""
Amazon 상품 페이지에서 kindle_data-v2.csv 책 설명(description) 수집.

productURL → Amazon 페이지 스크래핑 → #bookDescription_feature_div
체크포인트: data/kindle_desc_cache.json (500건마다)
API 키 불필요, 무료.

실행 예:
  python3 scripts/add_book_descriptions.py
  python3 scripts/add_book_descriptions.py --limit 500
  python3 scripts/add_book_descriptions.py --workers 5
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

CSV_PATH         = "data/kindle_data-v2.csv"
CACHE_PATH       = "data/kindle_desc_cache.json"
CHECKPOINT_EVERY = 500
REQUEST_DELAY    = 1.0   # Amazon 차단 방지

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SELECTORS = [
    "#bookDescription_feature_div",
    "#productDescription",
    '[data-feature-name="bookDescription"]',
]


def fetch_description(asin: str, url: str) -> str:
    """Amazon 상품 페이지에서 책 설명 추출."""
    try:
        time.sleep(REQUEST_DELAY)
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for selector in SELECTORS:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(separator=" ", strip=True)
                if len(text) > 50:
                    return text[:3000]
    except Exception:
        pass
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--workers", type=int, default=5,
                        help="동시 요청 수 (기본 5, Amazon 차단 방지를 위해 너무 높이지 말 것)")
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH, low_memory=False)
    full_df = df.copy()
    if args.limit:
        df = df.head(args.limit)

    cache: dict = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}

    remaining = [
        (str(row["asin"]), str(row["productURL"]))
        for _, row in df.iterrows()
        if str(row["asin"]) not in cache
    ]
    est_hours = len(remaining) * REQUEST_DELAY / args.workers / 3600
    print(f"전체: {len(df):,}개 | 캐시: {len(cache):,}개 | 남은 처리: {len(remaining):,}개")
    print(f"workers={args.workers} | 예상: 약 {est_hours:.1f}시간\n")

    success = sum(1 for v in cache.values() if v)
    processed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_description, asin, url): asin
            for asin, url in remaining
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping"):
            asin = futures[future]
            try:
                desc = future.result()
            except Exception:
                desc = ""

            cache[asin] = desc
            if desc:
                success += 1
            processed += 1

            if processed % CHECKPOINT_EVERY == 0:
                with open(CACHE_PATH, "w") as f:
                    json.dump(cache, f, ensure_ascii=False)
                print(f"  {processed:,}/{len(remaining):,} | 수집 성공: {success:,} ({success/processed*100:.0f}%)")

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, ensure_ascii=False)

    full_df["asin"] = full_df["asin"].astype(str)
    full_df["description"] = full_df["asin"].map(cache).fillna("")
    full_df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    filled = (full_df["description"].str.strip() != "").sum()
    print(f"\n완료! description 수집: {filled:,} / {len(full_df):,} ({filled/len(full_df)*100:.1f}%)")
    print(f"저장: {CSV_PATH}")


if __name__ == "__main__":
    main()
