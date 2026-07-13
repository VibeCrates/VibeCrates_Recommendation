"""
Prepares a unified training CSV from domain CSVs and query cache files.

Output CSV columns: item_id, content_text, image_path, query, domain

Usage:
  python scripts/prepare_dataset.py
  python scripts/prepare_dataset.py --domains movie music --output data/train_movie_music.csv
  python scripts/prepare_dataset.py --keep-no-query --drop-no-image
"""
import argparse
import json
import logging
import os

import pandas as pd

from src.data.preprocessing import DOMAIN_CONFIG, prepare_domain_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUERY_CACHE_PATHS = {
    "movie": "data/cache/query_cache_movie.json",
    "music": "data/cache/query_cache_music.json",
    "book":  "data/cache/query_cache_book.json",
}


def load_domain(domain: str, image_base_dir: str) -> pd.DataFrame:
    cfg = DOMAIN_CONFIG[domain]
    id_col = cfg["id_col"]

    logger.info(f"[{domain}] loading {cfg['csv']}")
    df = pd.read_csv(cfg["csv"], engine="python")
    logger.info(f"[{domain}] {len(df):,} rows loaded")

    cache_path = QUERY_CACHE_PATHS[domain]
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

        # Build a Series from validated cache entries only (skip None/empty values)
        valid_cache = pd.Series({k: v for k, v in cache.items() if v})
        cached = df[id_col].astype(str).map(valid_cache)

        orig = df["query"] if "query" in df.columns else pd.Series("", index=df.index)
        df["query"] = cached.fillna(orig)

        filled = cached.notna().sum()
        logger.info(f"[{domain}] query cache: {filled:,}/{len(df):,} filled from cache")
    else:
        logger.warning(f"[{domain}] query cache not found: {cache_path}")

    std_df = prepare_domain_df(domain, df, image_base_dir=image_base_dir)
    std_df["domain"] = domain
    return std_df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare unified training dataset from domain CSVs and query caches"
    )
    parser.add_argument(
        "--domains", nargs="+", default=["movie", "music", "book"],
        choices=["movie", "music", "book"],
        help="Domains to include (default: all three)"
    )
    parser.add_argument(
        "--output", default="data/train_dataset.csv",
        help="Output CSV path (default: data/train_dataset.csv)"
    )
    parser.add_argument(
        "--image-base-dir", default="data/images",
        help="Root directory for local images (default: data/images)"
    )
    parser.add_argument(
        "--keep-no-query", action="store_true",
        help="Keep rows with no query (dropped by default)"
    )
    parser.add_argument(
        "--drop-no-image", action="store_true",
        help="Drop rows with no image path (kept by default)"
    )
    args = parser.parse_args()

    dfs = []
    for domain in args.domains:
        df = load_domain(domain, args.image_base_dir)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined: {len(combined):,} rows across {len(args.domains)} domain(s)")

    if not args.keep_no_query:
        before = len(combined)
        combined = combined[
            combined["query"].notna() & combined["query"].str.strip().ne("")
        ]
        dropped = before - len(combined)
        if dropped:
            logger.info(f"Dropped {dropped:,} rows with no query → {len(combined):,} remaining")

    if args.drop_no_image:
        before = len(combined)
        combined = combined[combined["image_path"].ne("")]
        dropped = before - len(combined)
        if dropped:
            logger.info(f"Dropped {dropped:,} rows with no image → {len(combined):,} remaining")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    combined.to_csv(args.output, index=False)
    logger.info(f"Saved → {args.output}")

    print("\n=== Dataset Summary ===")
    for domain, count in combined.groupby("domain").size().items():
        print(f"  {domain:6s}: {count:>8,}")
    print(f"  {'total':6s}: {len(combined):>8,}")
    no_image = (combined["image_path"] == "").sum()
    no_query = (combined["query"].isna() | combined["query"].str.strip().eq("")).sum()
    if no_image:
        print(f"\n  WARNING: {no_image:,} rows have no image (will use HTTP fallback during training)")
    if no_query:
        print(f"  WARNING: {no_query:,} rows have no query")


if __name__ == "__main__":
    main()
