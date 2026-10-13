"""비교 시트(build_run_compare_sheet.py 출력)를 사람이 라벨링할 HTML 페이지로 만든다.

왜 필요한가:
  CSV/Numbers 시트는 판정기와 같은 조건(제목 가림, 텍스트만)으로 만들어졌다. 하지만 이
  시트는 판정기가 아니라 사람이 판정한다 — 사람은 제목과 표지를 보고 작품을 알아본다.
  그래서 같은 쌍·같은 좌우 배치(정답지 _key.csv가 그대로 유효)에 제목·원문·표지를 붙인다.

이미지:
  /Users/hyun/images는 옛 ID 체계라 일부만 맞는다(book은 kdl_ 접두사를 뗀 이름만).
  없으면 canonical CSV의 원본 URL, 그것도 죽었으면 영화는 TMDB(/find, .env의
  TMDB_API_KEY), 음악은 Deezer 검색(fetch_music_covers_deezer.py와 같은 일치 검사)에서
  받는다. 받은 이미지는 sips로 줄여 base64로 넣는다 —
  한 파일로 열리고, 깨진 링크가 라벨링 도중에 드러나지 않게.

한국어:
  {prefix}_ko.json(쿼리 번역 + 아이템별 줄거리·소개·장르 번역)이 있으면 한국어로 보여 주고
  원문은 접어 둔다. 사람 이름·제목은 원문 그대로다. 가사는 저작권 때문에 번역하지 않고
  내용 요지만 적었다.

라벨 저장은 scripts/serve_compare_page.py로 띄워서 연다 — 누를 때마다
{prefix}_hyun.csv(원래 시트와 같은 pair_id, query, 선택, 메모 열)에 바로 기록된다.

사용:
  python scripts/build_compare_page.py --prefix experiments/compare_titledrop_20261009
  python scripts/serve_compare_page.py --prefix experiments/compare_titledrop_20261009
"""
import argparse
import base64
import csv
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.fetch_music_covers_deezer import first_artist, norm
from scripts.judge_eval_relevance import build_item_lookup

IMAGE_ROOT = "/Users/hyun/images"
IMAGE_CACHE = "data/cache/compare_page_images.json"      # 다시 빌드할 때 네트워크를 다시 타지 않게
FIELD_RE = re.compile(r"^(Genre|Director|Cast|Overview|Artist|Album|Lyrics excerpt|Description|"
                      r"Author|Blurb|Category): (.*)$")
FIELD_KO = {"Genre": "장르", "Director": "감독", "Cast": "출연", "Overview": "줄거리",
            "Artist": "아티스트", "Album": "앨범", "Lyrics excerpt": "가사 요지",
            "Description": "곡 설명", "Author": "저자", "Blurb": "소개", "Category": "분류"}


def korean_body(body: str, ko: dict) -> str:
    """'라벨: 값' 줄을 한국어 라벨로 바꾸고, 번역이 있는 값은 번역으로 바꾼다."""
    fields = []
    for line in body.split("\n"):
        m = FIELD_RE.match(line)
        if m:
            fields.append([m.group(1), m.group(2)])
        elif fields:
            fields[-1][1] += "\n" + line
    out = []
    for f, v in fields:
        v = ko.get(f, v)
        if f == "Lyrics excerpt":
            v = v.removeprefix("[가사 요지] ")
        out.append(f"{FIELD_KO[f]}: {v}")
    return "\n".join(out)
WIDTH = 260


def local_image(domain: str, item_id: str) -> str | None:
    names = [item_id]
    if domain == "book" and item_id.startswith("kdl_"):
        names.append(item_id[4:])
    for n in names:
        p = os.path.join(IMAGE_ROOT, domain, f"{n}.jpg")
        if os.path.exists(p):
            return p
    return None


def image_urls() -> dict:
    urls = {}
    m = pd.read_csv("data/canonical/movie_canonical.csv", dtype={"imdbId": str}, usecols=["imdbId", "Poster"])
    urls.update({("movie", i): u for i, u in zip(m.imdbId, m.Poster) if isinstance(u, str) and u.startswith("http")})
    mu = pd.read_csv("data/canonical/music_canonical.csv", usecols=["id", "img"])
    urls.update({("music", i): u for i, u in zip(mu.id, mu.img) if isinstance(u, str) and u.startswith("http")})
    b = pd.read_csv("data/canonical/book_canonical_v2.csv", usecols=["asin", "imgUrl"])
    urls.update({("book", i): u for i, u in zip(b.asin, b.imgUrl) if isinstance(u, str) and u.startswith("http")})
    return urls


def tmdb_key() -> str | None:
    if os.environ.get("TMDB_API_KEY"):
        return os.environ["TMDB_API_KEY"]
    for line in open(".env", encoding="utf-8") if os.path.exists(".env") else []:
        if line.startswith("TMDB_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "VibeCrates/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def fallback_url(domain: str, item_id: str, music_meta: dict) -> str | None:
    try:
        if domain == "movie" and (key := tmdb_key()):
            res = get_json(f"https://api.themoviedb.org/3/find/tt{item_id.zfill(7)}"
                           f"?api_key={key}&external_source=imdb_id").get("movie_results") or []
            if res and res[0].get("poster_path"):
                return "https://image.tmdb.org/t/p/w342" + res[0]["poster_path"]
        if domain == "music" and item_id in music_meta:
            track, artist = music_meta[item_id]
            # 엄격 검색은 표기가 조금만 달라도 0건이라 느슨한 검색도 본다. 일치 검사는 같다.
            for q in (f'track:"{track}" artist:"{artist}"', f"{artist} {track}"):
                data = get_json("https://api.deezer.com/search?q=" + urllib.parse.quote(q)
                                + "&limit=1").get("data") or []
                if data and norm(data[0].get("title", "")) == norm(track) and \
                   norm((data[0].get("artist") or {}).get("name", "")) == norm(artist):
                    return (data[0].get("album") or {}).get("cover_big")
            res = get_json("https://itunes.apple.com/search?entity=song&limit=1&term="
                           + urllib.parse.quote(f"{artist} {track}")).get("results") or []
            if res and norm(res[0].get("trackName", "")) == norm(track) and \
               norm(res[0].get("artistName", "")) == norm(artist):
                return res[0]["artworkUrl100"].replace("100x100", "600x600")
    except Exception:
        pass
    return None


def to_data_uri(src: str, is_url: bool) -> str | None:
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        try:
            if is_url:
                req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as r, open(raw, "wb") as f:
                    f.write(r.read())
            else:
                raw = src
            out = os.path.join(tmp, "out.jpg")
            subprocess.run(["sips", "-s", "format", "jpeg", "--resampleWidth", str(WIDTH), raw, "--out", out],
                           check=True, capture_output=True)
            if os.path.getsize(out) < 1500:          # openlibrary 1x1 등 빈 이미지
                return None
            return "data:image/jpeg;base64," + base64.b64encode(open(out, "rb").read()).decode()
        except Exception:
            return None


def fetch_image(key, urls, music_meta) -> str | None:
    domain, item_id = key
    p = local_image(domain, item_id)
    if p and (uri := to_data_uri(p, False)):
        return uri
    if key in urls and (uri := to_data_uri(urls[key], True)):
        return uri
    if url := fallback_url(domain, item_id, music_meta):
        return to_data_uri(url, True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="시트 접두사 (… .csv, …_key.csv)")
    args = ap.parse_args()

    sheet = list(csv.DictReader(open(f"{args.prefix}.csv", encoding="utf-8-sig")))
    key = {r["pair_id"]: r for r in csv.DictReader(open(f"{args.prefix}_key.csv", encoding="utf-8-sig"))}

    def sides(pid):
        k = key[pid]
        a, b = k["a_item"].split(":", 1), k["b_item"].split(":", 1)
        return (b, a) if k["b_side"] == "1" else (a, b)   # (후보1, 후보2)

    need = {tuple(x) for r in sheet for x in sides(r["pair_id"])}
    lookup = build_item_lookup(False, need)                 # 제목 보임
    urls = image_urls()
    mu = pd.read_csv("data/canonical/music_canonical.csv", usecols=["id", "name", "artists"])
    music_meta = {i: (t, first_artist(a)) for i, t, a in zip(mu.id, mu.name, mu.artists)
                  if ("music", i) in need}
    cache = json.load(open(IMAGE_CACHE)) if os.path.exists(IMAGE_CACHE) else {}
    todo = [k for k in need if cache.get(f"{k[0]}:{k[1]}") is None]
    with ThreadPoolExecutor(8) as ex:
        for k, uri in zip(todo, ex.map(lambda k: fetch_image(k, urls, music_meta), todo)):
            cache[f"{k[0]}:{k[1]}"] = uri
    json.dump(cache, open(IMAGE_CACHE, "w"))
    imgs = {k: cache.get(f"{k[0]}:{k[1]}") for k in need}

    ko_path = f"{args.prefix}_ko.json"
    ko = json.load(open(ko_path, encoding="utf-8")) if os.path.exists(ko_path) else {"queries": {}, "items": {}}

    pairs = []
    for r in sheet:
        cands = []
        for d, i in sides(r["pair_id"]):
            text = lookup.get((d, i), ("", True))[0]
            lines = text.split("\n")
            title = lines[0].split(": ", 1)[-1] if lines else i
            body = "\n".join(lines[1:])
            tr = ko["items"].get(f"{d}|{title}")
            cands.append({"domain": d, "title": title, "img": imgs.get((d, i)),
                          "body": korean_body(body, tr) if tr else body, "orig": body if tr else ""})
        pairs.append({"id": r["pair_id"], "query": r["query"],
                      "query_ko": ko["queries"].get(r["query"], ""), "c": cands})

    out = f"{args.prefix}_page.html"
    tpl = (Path(__file__).parent / "compare_page_template.html").read_text(encoding="utf-8")
    page = (tpl.replace("__TITLE__", html.escape(Path(args.prefix).name))
               .replace("__STORE__", Path(args.prefix).name)
               .replace("__DATA__", json.dumps(pairs, ensure_ascii=False).replace("</", "<\\/")))
    Path(out).write_text(page, encoding="utf-8")

    n_img = sum(v is not None for v in imgs.values())
    print(f"쌍 {len(pairs)}개 → {out}")
    print(f"  이미지 {n_img}/{len(need)} (없는 것: {sorted(f'{d}:{i}' for (d, i), v in imgs.items() if v is None)})")
    print(f"  크기 {os.path.getsize(out) / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
