"""
instrumental 트랙(lyrics 없음)의 설명을 수집.
1차: Last.fm track.getInfo (wiki.summary)
2차: Wikipedia API 검색 (Last.fm 실패 시)
결과: music_canonical.csv에 description 컬럼 추가
체크포인트: data/cache/music_desc_cache.json (500건마다)
"""

import os
import re
import json
import time
import requests
import pandas as pd

LASTFM_API_KEY = os.environ["LASTFM_API_KEY"]
CHECKPOINT_PATH = "data/cache/music_desc_cache.json"
CHECKPOINT_EVERY = 500
REQUEST_DELAY = 0.25  # 초당 4req


def parse_artists(artists_str: str) -> list[str]:
    try:
        return json.loads(artists_str)
    except Exception:
        return [str(artists_str)]


def clean_lastfm_summary(text: str) -> str:
    # Last.fm summary 끝에 붙는 "<a href=...>Read more</a>" 제거
    text = re.sub(r'<a\s[^>]*>.*?</a>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def fetch_lastfm(track_name: str, artist: str) -> str | None:
    try:
        r = requests.get(
            "http://ws.audioscrobbler.com/2.0/",
            params={
                "method": "track.getInfo",
                "api_key": LASTFM_API_KEY,
                "artist": artist,
                "track": track_name,
                "format": "json",
                "autocorrect": 1,
            },
            timeout=10,
        )
        data = r.json()
        summary = data.get("track", {}).get("wiki", {}).get("summary", "").strip()
        if summary:
            cleaned = clean_lastfm_summary(summary)
            if len(cleaned) > 30:
                return cleaned
    except Exception:
        pass
    return None


def fetch_wikipedia(track_name: str, artist: str) -> str | None:
    query = f"{track_name} {artist} song"
    try:
        # 검색
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            },
            headers={"User-Agent": "VibeCrates/1.0"},
            timeout=10,
        )
        results = r.json().get("query", {}).get("search", [])
        if not results:
            return None

        title = results[0]["title"]
        # 첫 단락 추출
        r2 = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
            headers={"User-Agent": "VibeCrates/1.0"},
            timeout=10,
        )
        extract = r2.json().get("extract", "").strip()
        if len(extract) > 30:
            return extract
    except Exception:
        pass
    return None


def save_checkpoint(cache: dict):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(cache, f, ensure_ascii=False)


def main():
    df = pd.read_csv("data/canonical/music_canonical.csv", low_memory=False)

    no_lyrics_mask = df.lyrics.isna() | (df.lyrics.astype(str).str.strip().isin(["", "nan"]))
    target = df[no_lyrics_mask][["id", "name", "artists"]].copy()
    print(f"대상 트랙: {len(target):,}개")

    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            cache = json.load(f)
        print(f"체크포인트 로드: {len(cache):,}개")
    else:
        cache = {}

    remaining = target[~target["id"].isin(cache)].to_dict("records")
    total = len(remaining)
    print(f"남은 처리 수: {total:,}개 | 예상 시간: {total * REQUEST_DELAY / 60:.0f}~{total * REQUEST_DELAY * 2 / 60:.0f}분\n")

    lastfm_hit = sum(1 for v in cache.values() if v and v.get("source") == "lastfm")
    wiki_hit = sum(1 for v in cache.values() if v and v.get("source") == "wikipedia")
    no_hit = sum(1 for v in cache.values() if not v or not v.get("text"))

    for i, row in enumerate(remaining, 1):
        tid = row["id"]
        name = str(row["name"])
        artists = parse_artists(str(row["artists"]))
        artist = artists[0] if artists else ""

        time.sleep(REQUEST_DELAY)
        desc = fetch_lastfm(name, artist)
        source = "lastfm"

        if not desc:
            time.sleep(REQUEST_DELAY)
            desc = fetch_wikipedia(name, artist)
            source = "wikipedia"

        if desc:
            cache[tid] = {"text": desc, "source": source}
            if source == "lastfm":
                lastfm_hit += 1
            else:
                wiki_hit += 1
        else:
            cache[tid] = {"text": None, "source": None}
            no_hit += 1

        if i % CHECKPOINT_EVERY == 0:
            save_checkpoint(cache)
            print(f"  {i:,}/{total:,} | Last.fm: {lastfm_hit:,} | Wikipedia: {wiki_hit:,} | 없음: {no_hit:,}")

    save_checkpoint(cache)

    # description 컬럼 추가
    desc_map = {tid: v.get("text") for tid, v in cache.items()}
    if "description" not in df.columns:
        df["description"] = None

    df["description"] = df.apply(
        lambda row: desc_map.get(row["id"], row.get("description"))
        if (pd.isna(row.get("lyrics")) or str(row.get("lyrics", "")).strip() in ["", "nan"])
        else row.get("description"),
        axis=1,
    )

    df.to_csv("data/canonical/music_canonical.csv", index=False)

    total_desc = df["description"].notna().sum()
    print(f"\n완료!")
    print(f"Last.fm 수집: {lastfm_hit:,}개 | Wikipedia 수집: {wiki_hit:,}개 | 미수집: {no_hit:,}개")
    print(f"description 컬럼 채워진 행: {total_desc:,}개 / {len(df):,}개")
    print(f"저장: data/canonical/music_canonical.csv")


if __name__ == "__main__":
    main()
