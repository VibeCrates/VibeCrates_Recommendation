"""
iTunes Search API로 img='no' 또는 null인 트랙의 앨범 아트 URL을 수집.
API 키 불필요. 아티스트명 + 트랙명으로 검색.
체크포인트: data/cache/music_img_cache.json (500건마다)
"""

import json
import time
import requests
import pandas as pd
import os

CHECKPOINT_PATH = "data/cache/music_img_cache.json"
CHECKPOINT_EVERY = 500
REQUEST_DELAY = 0.1  # 초당 10req


def parse_artists(artists_str: str) -> list[str]:
    try:
        return json.loads(artists_str)
    except Exception:
        return [str(artists_str)]


def fetch_artwork(track_name: str, artist: str) -> str | None:
    try:
        r = requests.get(
            "https://itunes.apple.com/search",
            params={
                "term": f"{track_name} {artist}",
                "media": "music",
                "entity": "song",
                "limit": 1,
            },
            headers={"User-Agent": "VibeCrates/1.0"},
            timeout=10,
        )
        results = r.json().get("results", [])
        if not results:
            return None
        url = results[0].get("artworkUrl100", "")
        if url:
            # 100x100 → 600x600으로 업스케일
            return url.replace("100x100bb", "600x600bb")
        return None
    except Exception:
        return None


def save_checkpoint(cache: dict):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(cache, f, ensure_ascii=False)


def main():
    df = pd.read_csv("data/canonical/music_canonical.csv", low_memory=False)
    no_img_mask = df.img.isna() | (df.img == "no")
    target = df[no_img_mask][["id", "name", "artists"]].copy()
    print(f"이미지 없는 트랙: {len(target):,}개", flush=True)

    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            cache = json.load(f)
        print(f"체크포인트 로드: {len(cache):,}개", flush=True)
    else:
        cache = {}

    remaining = target[~target["id"].isin(cache)].to_dict("records")
    total = len(remaining)
    eta = total * REQUEST_DELAY / 60
    print(f"남은 처리 수: {total:,}개 | 예상 시간: {eta:.0f}분\n", flush=True)

    found = sum(1 for v in cache.values() if v)
    not_found = len(cache) - found

    for i, row in enumerate(remaining, 1):
        tid = row["id"]
        name = str(row["name"])
        artists = parse_artists(str(row["artists"]))
        artist = artists[0] if artists else ""

        time.sleep(REQUEST_DELAY)
        url = fetch_artwork(name, artist)
        cache[tid] = url

        if url:
            found += 1
        else:
            not_found += 1

        if i % CHECKPOINT_EVERY == 0:
            save_checkpoint(cache)
            print(f"  {i:,}/{total:,} | 발견: {found:,}개 | 없음: {not_found:,}개", flush=True)

    save_checkpoint(cache)

    # CSV 업데이트
    desc_map = cache
    df["img"] = df.apply(
        lambda row: desc_map.get(row["id"], row["img"])
        if (pd.isna(row["img"]) or row["img"] == "no")
        else row["img"],
        axis=1,
    )
    df["img"] = df["img"].fillna("no")
    df.to_csv("data/canonical/music_canonical.csv", index=False)

    total_img = (df.img != "no").sum()
    print(f"\n완료!")
    print(f"iTunes 수집: {found:,}개 | 미수집: {not_found:,}개")
    print(f"전체 이미지 있음: {total_img:,}개 / {len(df):,}개")
    print(f"저장: data/canonical/music_canonical.csv")


if __name__ == "__main__":
    main()
