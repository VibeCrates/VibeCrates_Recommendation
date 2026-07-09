"""
example.png(콜라주 "vibe" 무드보드 밈) 레이아웃을 참고한 VibeCrates 추천 결과
목업 생성 스크립트.

정적 프론트엔드 소스(scripts/vibe_crate/index.html — 레이아웃/hover 카드 로직)는
그대로 두고, 이 스크립트는 데이터 계약에 맞춰 두 가지만 만든다:
  1. data.js  — RECOMMENDATIONS 배열 (id/title/domain/year/image/url)
     실제 데이터셋(MovieGenre_enriched.csv / music_features.csv /
     kindle_data-v2.csv)에서 직접 조회, 하드코딩 없음.
  2. images/  — /Users/hyun/images에서 실제 표지 이미지를 리사이즈해 복사

출력 폴더(index.html + data.js + images/)는 코드와 이미지 파일만으로
브라우저에서 바로 열림 (서버/빌드 불필요, data.js는 <script src>로
로드되므로 file:// 에서도 fetch CORS 문제 없음).

데이터 계약:
  {
    "id": "114709",           // 비노출, 이미지 파일 매핑 + DOM key 용도
    "title": "Toy Story",
    "domain": "movie",
    "year": 1995,             // 결측 시 null (프론트에서 "xxxx"로 표시)
    "image": "images/movie/114709.jpg",
    "url": "http://www.imdb.com/title/tt0114709"
  }

실행:
  python3 scripts/build_vibe_mockup.py
  python3 scripts/build_vibe_mockup.py --out mockups/vibe_crate --width 400
"""
import argparse
import json
import os
import shutil
from io import BytesIO

import pandas as pd
from PIL import Image

IMAGE_ROOT = "/Users/hyun/images"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "vibe_crate")
STATIC_SRC = os.path.join(STATIC_DIR, "index.html")
README_SRC = os.path.join(STATIC_DIR, "README.md")
DEFAULT_OUTPUT = "mockups/vibe_crate"
DEFAULT_TARGET_WIDTH = 400  # px

MOVIE_CSV = "data/MovieGenre_enriched.csv"
MUSIC_CSV = "data/music_features.csv"
BOOK_CSV  = "data/kindle_data-v2.csv"

# 데모에 노출할 (도메인, id) 목록. id는 movie=imdbId / music=spotify id / book=asin.
SELECTION = [
    ("music", "31AOj9sFz2gM0O3hMARRBx"),
    ("music", "6L89mwZXSOwYl76YXfX13s"),
    ("movie", "1798709"),
    ("movie", "338013"),
    ("book",  "B003XT603Q"),
    ("book",  "B000SEGHT6"),
    ("music", "6ADSaE87h8Y3lccZlBJdXH"),
    ("book",  "B0CD2CQZJR"),
    ("book",  "B09Q2F1X14"),
    ("music", "3d9DChrdc6BOeFsbrZ3Is0"),
    ("music", "003vvx7Niy0yvhvHt4a68B"),
    ("movie", "117951"),
]


def load_lookup_tables():
    movie = pd.read_csv(MOVIE_CSV, low_memory=False)
    movie["imdbId"] = movie["imdbId"].astype(str)
    movie = movie.set_index("imdbId")

    music = pd.read_csv(MUSIC_CSV, low_memory=False, usecols=["id", "name", "year"])
    music = music.set_index("id")

    book = pd.read_csv(BOOK_CSV, low_memory=False,
                        usecols=["asin", "title", "publishedDate", "productURL"])
    book = book.set_index("asin")

    return movie, music, book


def build_item(domain: str, file_id: str, tables) -> dict:
    movie, music, book = tables

    if domain == "movie":
        row = movie.loc[file_id]
        release_date = row["release_date"]
        year = int(release_date[:4]) if isinstance(release_date, str) and release_date else None
        return dict(id=file_id, domain=domain, title=row["Title"], year=year,
                    url=row["Imdb Link"])

    if domain == "music":
        row = music.loc[file_id]
        year = int(row["year"]) if pd.notna(row["year"]) else None
        return dict(id=file_id, domain=domain, title=row["name"], year=year,
                    url=f"https://open.spotify.com/track/{file_id}")

    if domain == "book":
        row = book.loc[file_id]
        published = row["publishedDate"]
        year = int(published[:4]) if isinstance(published, str) and published else None
        return dict(id=file_id, domain=domain, title=row["title"], year=year,
                    url=row["productURL"])

    raise ValueError(f"unknown domain: {domain}")


def copy_resized_image(domain: str, file_id: str, out_dir: str, target_width: int) -> str:
    src_path = os.path.join(IMAGE_ROOT, domain, f"{file_id}.jpg")
    rel_path = os.path.join("images", domain, f"{file_id}.jpg")
    dst_path = os.path.join(out_dir, rel_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    im = Image.open(src_path).convert("RGB")
    ratio = target_width / im.width
    im = im.resize((target_width, int(im.height * ratio)), Image.LANCZOS)
    im.save(dst_path, format="JPEG", quality=82, optimize=True)

    return rel_path.replace(os.sep, "/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="출력 폴더 경로")
    parser.add_argument("--width", type=int, default=DEFAULT_TARGET_WIDTH, help="이미지 리사이즈 목표 너비(px)")
    args = parser.parse_args()

    tables = load_lookup_tables()
    os.makedirs(args.out, exist_ok=True)

    items = []
    for domain, file_id in SELECTION:
        item = build_item(domain, file_id, tables)
        item["image"] = copy_resized_image(domain, file_id, args.out, args.width)
        items.append(item)

    # var를 써야 최상위 선언이 window.RECOMMENDATIONS로 붙는다
    # (const/let 최상위 선언은 window 프로퍼티가 되지 않아 index.html에서 못 읽음)
    data_js = "var RECOMMENDATIONS = " + json.dumps(items, ensure_ascii=False, indent=2) + ";\n"
    with open(os.path.join(args.out, "data.js"), "w", encoding="utf-8") as f:
        f.write(data_js)

    shutil.copy(STATIC_SRC, os.path.join(args.out, "index.html"))
    shutil.copy(README_SRC, os.path.join(args.out, "README.md"))

    missing_year = sum(1 for it in items if it["year"] is None)
    print(f"아이템 {len(items)}개 처리 완료 (year 결측 {missing_year}개 → 프론트에서 'xxxx' 표시)")
    print(f"저장 → {args.out}/index.html, {args.out}/data.js, {args.out}/images/")


if __name__ == "__main__":
    main()
