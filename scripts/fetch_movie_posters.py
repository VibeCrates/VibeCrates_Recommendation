"""
포스터가 죽은 movie 항목의 커버를 TMDB에서 복구한다.

왜 필요한가:
  MovieGenre 데이터셋의 `Poster` URL은 images-na.ssl-images-amazon.com을 가리키는데
  상당수가 만료돼 404(본문 9바이트)를 돌려준다 — 실측 39,383건 중 10,607건(26.9%) 실패.
  그대로 두면 movie의 27%가 포스터 없이 합성되고(generate_item_descriptions의
  grounding_noimg 분기), 학습 시 z_image도 빈다. book이 ~97%라 도메인 간 불균형이 된다.

기존 `fetch_movie_meta.py`는 release_date/running_time/director/actor 4개만 캐시에
남기고 poster_path를 버려서 재사용할 수 없다. 기존 캐시(39,516건)를 건드리지 않도록
별도 캐시(data/cache/tmdb_poster_cache.json)에 쓴다.

대상 판정은 CSV의 URL이 아니라 **로컬 이미지 파일 존재 여부**로 한다 — download_images가
이미 404·1KB미만을 걸러낸 뒤이므로, 파일이 없다는 것이 곧 "확보 실패"다.

사용:
  TMDB_API_KEY="..." python3 scripts/fetch_movie_posters.py
  TMDB_API_KEY="..." python3 scripts/fetch_movie_posters.py --limit 100
"""

import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

CSV_PATH = "data/canonical/movie_canonical.csv"
IMAGE_DIR = Path("data/images/movie")
CACHE_PATH = "data/cache/tmdb_poster_cache.json"
CHECKPOINT_EVERY = 500

# w500은 Qwen2.5-VL 입력으로 충분하고 원본(original)보다 훨씬 가볍다.
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

API_KEY = os.environ["TMDB_API_KEY"]

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json", "User-Agent": "VibeCrates/1.0"})


def find_poster_path(imdb_id: str) -> str:
    """IMDB ID → TMDB /find → poster_path. 없으면 빈 문자열."""
    try:
        r = SESSION.get(
            f"https://api.themoviedb.org/3/find/tt{str(imdb_id).zfill(7)}",
            params={"api_key": API_KEY, "external_source": "imdb_id"},
            timeout=10,
        )
        if r.status_code != 200:
            return ""
        results = r.json().get("movie_results") or []
        return results[0].get("poster_path") or "" if results else ""
    except requests.RequestException:
        return ""


def download(imdb_id: str, poster_path: str) -> bool:
    out = IMAGE_DIR / f"{imdb_id}.jpg"
    if out.exists():
        return True
    try:
        r = SESSION.get(TMDB_IMAGE_BASE + poster_path, timeout=15)
        r.raise_for_status()
        # download_images.py와 같은 하한 — 플레이스홀더 이미지를 걸러낸다.
        if len(r.content) < 1024:
            return False
        out.write_bytes(r.content)
        return True
    except requests.RequestException:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=10, help="TMDB 한도(약 40 req/s) 여유")
    args = ap.parse_args()

    df = pd.read_csv(CSV_PATH, low_memory=False)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}

    ids = [str(i) for i in df["imdbId"].tolist()]
    todo = [i for i in ids if not (IMAGE_DIR / f"{i}.jpg").exists() and i not in cache]
    if args.limit:
        todo = todo[: args.limit]
    print(f"전체 {len(ids):,} / 이미지 없음 {sum(1 for i in ids if not (IMAGE_DIR / f'{i}.jpg').exists()):,} "
          f"/ 조회 대상 {len(todo):,} (캐시 {len(cache):,})")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(find_poster_path, i): i for i in todo}
        for n, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc="TMDB find"), 1):
            cache[futures[fut]] = fut.result()
            if n % CHECKPOINT_EVERY == 0:
                json.dump(cache, open(CACHE_PATH, "w"))
    json.dump(cache, open(CACHE_PATH, "w"))

    found = {i: p for i, p in cache.items() if p and not (IMAGE_DIR / f"{i}.jpg").exists()}
    print(f"poster_path 확보 {sum(1 for p in cache.values() if p):,} / 조회 {len(cache):,} "
          f"→ 다운로드 대상 {len(found):,}")

    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers * 3) as ex:
        futures = [ex.submit(download, i, p) for i, p in found.items()]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="download"):
            ok += bool(fut.result())

    have = sum(1 for i in ids if (IMAGE_DIR / f"{i}.jpg").exists())
    print(f"복구 {ok:,}장 — movie 이미지 커버리지 {have:,}/{len(ids):,} ({have / len(ids) * 100:.1f}%)")


if __name__ == "__main__":
    main()
