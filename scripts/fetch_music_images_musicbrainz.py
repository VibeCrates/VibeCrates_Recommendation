"""
img='no'인 음악 트랙의 앨범 아트 URL을 MusicBrainz + Cover Art Archive로 수집.
API 키 불필요. MusicBrainz rate limit: 1 req/sec.

동작 순서:
  1. MusicBrainz Search API로 트랙명 + 아티스트 검색 → release MBID 획득
  2. Cover Art Archive API로 해당 release의 이미지 URL 조회
  3. music_canonical.csv의 img 컬럼 업데이트

체크포인트: data/cache/musicbrainz_cache.json (500건마다)

실행 예:
  python3 scripts/fetch_music_images_musicbrainz.py
  python3 scripts/fetch_music_images_musicbrainz.py --limit 1000  # 일부만 테스트
"""

import os
import json
import time
import argparse

import requests
import pandas as pd
from tqdm import tqdm

CHECKPOINT_PATH = "data/cache/musicbrainz_cache.json"
CHECKPOINT_EVERY = 500
REQUEST_DELAY = 1.1  # MusicBrainz 1 req/sec 준수

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "VibeCrates/1.0 (github.com/VibeCrates)"})


def search_release_mbid(track_name: str, artist: str) -> str | None:
    """MusicBrainz에서 트랙명 + 아티스트로 release MBID 검색."""
    try:
        r = SESSION.get(
            "https://musicbrainz.org/ws/2/recording",
            params={
                "query": f'recording:"{track_name}" AND artist:"{artist}"',
                "fmt": "json",
                "limit": 1,
            },
            timeout=10,
        )
        r.raise_for_status()
        recordings = r.json().get("recordings", [])
        if not recordings:
            return None
        releases = recordings[0].get("releases", [])
        if not releases:
            return None
        return releases[0]["id"]
    except Exception:
        return None


def fetch_cover_art_url(mbid: str) -> str | None:
    """Cover Art Archive에서 release MBID로 이미지 URL 조회."""
    try:
        r = SESSION.get(
            f"https://coverartarchive.org/release/{mbid}",
            timeout=10,
            allow_redirects=True,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        images = r.json().get("images", [])
        for img in images:
            if img.get("front"):
                thumbnails = img.get("thumbnails", {})
                return thumbnails.get("500") or thumbnails.get("large") or img.get("image")
        if images:
            img = images[0]
            thumbnails = img.get("thumbnails", {})
            return thumbnails.get("500") or thumbnails.get("large") or img.get("image")
    except Exception:
        return None
    return None


def parse_artists(artists_str: str) -> str:
    try:
        artists = json.loads(artists_str)
        return artists[0] if artists else ""
    except Exception:
        return str(artists_str)


def save_checkpoint(cache: dict):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(cache, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="처리할 최대 건수 (테스트용)")
    args = parser.parse_args()

    df = pd.read_csv("data/canonical/music_canonical.csv", low_memory=False)

    no_img_mask = df["img"].isna() | (df["img"].astype(str).str.strip().isin(["no", "nan", ""]))
    target = df[no_img_mask][["id", "name", "artists"]].copy()

    if args.limit:
        target = target.head(args.limit)

    print(f"대상 트랙: {len(target):,}개")
    print(f"예상 소요 시간: 약 {len(target) * REQUEST_DELAY / 3600:.1f}시간\n")

    cache = json.load(open(CHECKPOINT_PATH)) if os.path.exists(CHECKPOINT_PATH) else {}
    remaining = target[~target["id"].isin(cache)].to_dict("records")
    print(f"체크포인트 로드: {len(cache):,}개 | 남은 처리: {len(remaining):,}개\n")

    mbid_hit = 0
    art_hit = 0
    no_hit = 0

    for i, row in enumerate(tqdm(remaining), 1):
        track_id = str(row["id"])
        artist = parse_artists(str(row["artists"]))
        track_name = str(row["name"])

        time.sleep(REQUEST_DELAY)
        mbid = search_release_mbid(track_name, artist)

        if mbid:
            mbid_hit += 1
            time.sleep(REQUEST_DELAY)
            url = fetch_cover_art_url(mbid)
        else:
            url = None

        if url:
            art_hit += 1
            cache[track_id] = url
        else:
            no_hit += 1
            cache[track_id] = None

        if i % CHECKPOINT_EVERY == 0:
            save_checkpoint(cache)
            print(f"  {i:,}/{len(remaining):,} | MBID 발견: {mbid_hit:,} | 이미지 URL: {art_hit:,} | 미수집: {no_hit:,}")

    save_checkpoint(cache)

    # img 컬럼 업데이트
    url_map = {tid: url for tid, url in cache.items() if url}
    df["id"] = df["id"].astype(str)
    update_mask = no_img_mask & df["id"].isin(url_map)
    df.loc[update_mask, "img"] = df.loc[update_mask, "id"].map(url_map)
    df.to_csv("data/canonical/music_canonical.csv", index=False)

    updated = update_mask.sum()
    total_with_img = (df["img"].notna() & ~df["img"].isin(["no", "nan", ""])).sum()
    print(f"\n완료!")
    print(f"MBID 발견: {mbid_hit:,} | 이미지 URL 수집: {art_hit:,} | 미수집: {no_hit:,}")
    print(f"img 컬럼 업데이트: {updated:,}개")
    print(f"전체 이미지 보유: {total_with_img:,} / {len(df):,}개")
    print(f"저장: data/canonical/music_canonical.csv")


if __name__ == "__main__":
    main()
