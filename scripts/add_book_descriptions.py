"""
Amazon 상품 페이지에서 kindle_data-v2.csv 책 설명(description) 수집.

- 실패 항목은 MAX_RETRIES까지 자동 재시도
- MAX_RETRIES 초과 시 영구 스킵 → 반드시 종료됨
- 샤딩: --shard / --num-shards 로 여러 프로세스 병렬 실행 가능
- 병합: --merge 로 모든 샤드 캐시 합쳐서 CSV 업데이트

실행 예:
  # 단일 프로세스
  python scripts/add_book_descriptions.py

  # 4개 프로세스 병렬 (터미널 4개 또는 nohup)
  python scripts/add_book_descriptions.py --shard 0 --num-shards 4
  python scripts/add_book_descriptions.py --shard 1 --num-shards 4
  python scripts/add_book_descriptions.py --shard 2 --num-shards 4
  python scripts/add_book_descriptions.py --shard 3 --num-shards 4

  # 모든 샤드 완료 후 CSV 업데이트
  python scripts/add_book_descriptions.py --merge --num-shards 4
"""
import argparse
import glob
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
REQUEST_DELAY    = 1.0
MAX_RETRIES      = 3

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


def shard_cache_path(shard: int) -> str:
    return f"data/kindle_desc_cache_s{shard}.json"


def fetch_description(asin: str, url: str) -> str:
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


def load_cache(path: str) -> dict:
    """캐시 로드. 구버전 str 포맷 → {"desc": str, "tries": int} 마이그레이션."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for asin, val in raw.items():
        if isinstance(val, dict):
            result[asin] = val
        else:
            result[asin] = {"desc": val if isinstance(val, str) else "", "tries": 1}
    return result


def save_cache(cache: dict, path: str):
    with open(path, "w") as f:
        json.dump(cache, f, ensure_ascii=False)


def run_shard(df: pd.DataFrame, shard: int, num_shards: int, max_retries: int, workers: int):
    cache_path = shard_cache_path(shard) if num_shards > 1 else CACHE_PATH
    cache = load_cache(cache_path)

    # 이 샤드가 담당하는 행만 추출
    shard_df = df.iloc[shard::num_shards].reset_index(drop=True)

    remaining = [
        (str(row["asin"]), str(row["productURL"]))
        for _, row in shard_df.iterrows()
        if not cache.get(str(row["asin"]), {}).get("desc")
        and cache.get(str(row["asin"]), {}).get("tries", 0) < max_retries
    ]

    n_success = sum(1 for v in cache.values() if v.get("desc"))
    n_giveup  = sum(1 for v in cache.values() if not v.get("desc") and v.get("tries", 0) >= max_retries)

    shard_label = f"[shard {shard}/{num_shards-1}]" if num_shards > 1 else ""
    print(f"{shard_label} 담당: {len(shard_df):,} | 성공: {n_success:,} | 포기: {n_giveup:,} | 이번 처리: {len(remaining):,}")

    if not remaining:
        print(f"{shard_label} 처리할 항목 없음. 종료.")
        return cache, cache_path

    est_hours = len(remaining) * REQUEST_DELAY / workers / 3600
    print(f"{shard_label} workers={workers} | 예상: 약 {est_hours:.1f}시간\n")

    processed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_description, asin, url): asin
            for asin, url in remaining
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Scraping{shard_label}"):
            asin = futures[future]
            try:
                desc = future.result()
            except Exception:
                desc = ""

            prev_tries = cache.get(asin, {}).get("tries", 0)
            cache[asin] = {"desc": desc, "tries": prev_tries + 1}
            if desc:
                n_success += 1
            processed += 1

            if processed % CHECKPOINT_EVERY == 0:
                save_cache(cache, cache_path)
                n_giveup_now = sum(1 for v in cache.values() if not v.get("desc") and v.get("tries", 0) >= max_retries)
                print(f"  {shard_label} {processed:,}/{len(remaining):,} | 성공: {n_success:,} | 포기: {n_giveup_now:,}")

    save_cache(cache, cache_path)
    return cache, cache_path


def run_merge(df: pd.DataFrame, num_shards: int):
    """모든 샤드 캐시 합산 후 CSV 업데이트."""
    merged: dict = load_cache(CACHE_PATH)  # 기존 메인 캐시 베이스

    shard_files = sorted(glob.glob("data/kindle_desc_cache_s*.json"))
    if not shard_files:
        print("샤드 캐시 파일 없음. 메인 캐시만으로 CSV 업데이트.")
    else:
        for path in shard_files:
            shard_cache = load_cache(path)
            for asin, val in shard_cache.items():
                existing = merged.get(asin)
                # 성공한 값 우선, 아니면 시도 횟수 높은 것으로
                if not existing or (not existing.get("desc") and val.get("desc")):
                    merged[asin] = val
                elif not existing.get("desc") and not val.get("desc"):
                    merged[asin] = {"desc": "", "tries": max(existing.get("tries", 0), val.get("tries", 0))}
            print(f"  병합: {path} ({len(shard_cache):,}개)")

    save_cache(merged, CACHE_PATH)

    df["description"] = df["asin"].map(lambda a: merged.get(a, {}).get("desc", "") or "")
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    filled   = (df["description"].str.strip() != "").sum()
    n_giveup = sum(1 for v in merged.values() if not v.get("desc") and v.get("tries", 0) >= MAX_RETRIES)
    print(f"\n병합 완료! description: {filled:,} / {len(df):,} ({filled/len(df)*100:.1f}%)")
    print(f"포기 항목: {n_giveup:,} | 저장: {CSV_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard",       type=int, default=0)
    parser.add_argument("--num-shards",  type=int, default=1,
                        help="병렬 프로세스 수 (기본 1 = 샤딩 없음)")
    parser.add_argument("--merge",       action="store_true",
                        help="모든 샤드 캐시 합산 후 CSV 업데이트")
    parser.add_argument("--workers",     type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--limit",       type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["asin"] = df["asin"].astype(str)
    if args.limit:
        df = df.head(args.limit)

    if args.merge:
        run_merge(df, args.num_shards)
        return

    cache, cache_path = run_shard(df, args.shard, args.num_shards, args.max_retries, args.workers)

    # 단일 프로세스면 바로 CSV 업데이트
    if args.num_shards == 1:
        df["description"] = df["asin"].map(lambda a: cache.get(a, {}).get("desc", "") or "")
        df.to_csv(CSV_PATH, index=False, encoding="utf-8")
        filled = (df["description"].str.strip() != "").sum()
        print(f"\n완료! description: {filled:,} / {len(df):,} ({filled/len(df)*100:.1f}%)")
        print(f"저장: {CSV_PATH}")
    else:
        print(f"\n샤드 {args.shard} 완료. 모든 샤드 종료 후 --merge 실행.")


if __name__ == "__main__":
    main()
