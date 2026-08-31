#!/usr/bin/env python
"""표지 없는 music 트랙의 앨범 아트를 Deezer에서 수집해 내려받는다.

왜 Deezer인가 (2026-08-31 파일럿, 표본 100곡)
  Deezer 87% 히트 · 응답 전부 200 · 차단 없음
  iTunes 53% 히트 · 100건 중 44건이 429/403 차단
  iTunes는 차단되지 않은 56건 중 53건이 히트였다. 즉 세션 3의 0.9%는 커버가 없어서가
  아니라 차단당해서였고, 지금도 여전히 차단한다.
  Spotify 공식 API는 우리 id로 바로 조회할 수 있어 원래 가장 확실하지만, 배치 403과
  23시간 rate limit에 막힌 이력이 있고 자격증명도 없다.

이름으로 검색하는 방식이라 엉뚱한 곡이 붙을 수 있다. 그래서 받은 곡의 제목·아티스트를
정규화해 대조하고, **둘 다 어긋나면 버린다**. 파일럿에서 히트 52곡 중 49곡이 완전 일치,
3곡이 표기 차이, 완전 불일치는 0건이었다.

내려받은 뒤에도 검사한다. 1KB 미만이거나 JPEG/PNG 헤더가 아니면 버린다 — Open Library가
커버 없는 책에 43바이트 빈 이미지를 주는 바람에 그 책들이 전부 같은 z_image를 갖게 될
뻔한 사고가 있었다(세션 17).

체크포인트를 500건마다 저장하므로 중간에 끊겨도 이어받는다.

사용:
  python scripts/fetch_music_covers_deezer.py \
      --image-dir data/images/music --canonical data/canonical/music_canonical.csv
"""
import argparse
import json
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

API = "https://api.deezer.com/search"
MIN_BYTES = 1024


def norm(text: str) -> str:
    """비교용 정규화. 괄호·하이픈 뒤 꼬리(feat., - Remaster 등)를 떼고 기호를 없앤다."""
    s = unicodedata.normalize("NFKD", str(text)).lower()
    s = re.sub(r"\s*[\(\[-].*$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def first_artist(raw) -> str:
    try:
        parsed = json.loads(raw)
        return parsed[0] if parsed else ""
    except Exception:
        return str(raw)


class Fetcher:
    def __init__(self, image_dir: str, delay: float):
        self.image_dir = image_dir
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "VibeCrates/1.0 (research)"})
        self.lock = threading.Lock()
        self.stats = {"ok": 0, "no_result": 0, "mismatch": 0, "bad_image": 0, "error": 0}

    def bump(self, key: str) -> None:
        with self.lock:
            self.stats[key] += 1

    def run_one(self, item_id: str, track: str, artist: str) -> str:
        query = f'track:"{track}" artist:"{artist}"'
        try:
            r = self.session.get(API, params={"q": query, "limit": 1}, timeout=15)
            if r.status_code != 200:
                time.sleep(5)                      # 과속 신호면 물러선다
                self.bump("error")
                return "error"
            data = r.json().get("data") or []
        except Exception:
            self.bump("error")
            return "error"

        if not data:
            self.bump("no_result")
            return "no_result"

        hit = data[0]
        url = (hit.get("album") or {}).get("cover_xl")
        if not url:
            self.bump("no_result")
            return "no_result"

        # 제목·아티스트가 둘 다 어긋나면 다른 곡이다
        if norm(hit.get("title", "")) != norm(track) and \
           norm((hit.get("artist") or {}).get("name", "")) != norm(artist):
            self.bump("mismatch")
            return "mismatch"

        try:
            img = self.session.get(url, timeout=30).content
        except Exception:
            self.bump("error")
            return "error"

        if len(img) < MIN_BYTES or img[:3] not in (b"\xff\xd8\xff", b"\x89PN"):
            self.bump("bad_image")
            return "bad_image"

        path = os.path.join(self.image_dir, f"{item_id}.jpg")
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(img)
        os.replace(tmp, path)                      # 부분 파일이 정상으로 보이지 않게
        self.bump("ok")
        return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", default="data/images/music")
    ap.add_argument("--canonical", default="data/canonical/music_canonical.csv")
    ap.add_argument("--checkpoint", default="data/cache/deezer_cover_cache.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--delay", type=float, default=0.25, help="요청 간 간격(초)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="이전 실행의 error 건을 다시 시도한다")
    args = ap.parse_args()

    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)

    done: dict[str, str] = {}
    if os.path.exists(args.checkpoint):
        with open(args.checkpoint) as f:
            done = json.load(f)
        if args.retry_failed:
            done = {k: v for k, v in done.items() if v != "error"}
        print(f"체크포인트 {len(done):,}건 이어받음")

    df = pd.read_csv(args.canonical, usecols=["id", "name", "artists"], low_memory=False)
    df = df.drop_duplicates("id")
    have = {n[:-4] for n in os.listdir(args.image_dir) if n.endswith(".jpg")}
    todo = df[~df["id"].isin(have) & ~df["id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"전체 {len(df):,}곡 · 표지 보유 {len(have):,} · 이번 대상 {len(todo):,}")

    fetcher = Fetcher(args.image_dir, args.delay)
    started = time.time()
    counter = {"n": 0}

    def work(row):
        track = re.sub(r"\s*\(feat\..*?\)", "", str(row.name_)).strip()
        result = fetcher.run_one(row.id, track, first_artist(row.artists))
        with fetcher.lock:
            done[row.id] = result
            counter["n"] += 1
            n = counter["n"]
        if n % 500 == 0:
            with fetcher.lock, open(args.checkpoint, "w") as f:
                json.dump(done, f)
            rate = n / max(time.time() - started, 1)
            left = (len(todo) - n) / max(rate, 1e-9) / 60
            print(f"  {n:,}/{len(todo):,}  {fetcher.stats}  {rate:.1f}건/초  잔여 {left:.0f}분",
                  flush=True)
        time.sleep(args.delay)

    rows = list(todo.rename(columns={"name": "name_"}).itertuples(index=False))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, rows))

    with open(args.checkpoint, "w") as f:
        json.dump(done, f)

    total = max(len(todo), 1)
    print(f"\n완료 {len(todo):,}건 · {(time.time()-started)/60:.0f}분")
    for k, v in fetcher.stats.items():
        print(f"  {k:10s} {v:>7,}  ({v/total:.1%})")


if __name__ == "__main__":
    main()
