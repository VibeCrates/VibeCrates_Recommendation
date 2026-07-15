"""
example.png(콜라주 "vibe" 무드보드 밈) 레이아웃을 참고한 VibeCrates "보드" 목업
생성 스크립트.

정적 프론트엔드 소스(scripts/vibe_crate/main.html, board.html, style.css,
common.js — 레이아웃/인터랙션 로직)는 그대로 두고, 이 스크립트는 두 가지만 만든다:
  1. boards.js — BOARDS 배열 (보드 여러 개, 각 보드는 아이템 배열을 가짐)
     아이템의 title/year/url은 실제 데이터셋(movie_canonical.csv /
     music_canonical.csv / book_canonical.csv)에서 직접 조회, 하드코딩 없음.
  2. images/  — /Users/hyun/images에서 실제 표지 이미지를 리사이즈해 복사

출력 폴더(main.html + board.html + boards.js + images/ ...)는 코드와 이미지
파일만으로 브라우저에서 바로 열림 (서버/빌드 불필요). 단, board.html의
"캡쳐하기" 버튼만 html2canvas를 CDN에서 불러오므로 그 기능만 인터넷 연결이 필요.

아이템 데이터 계약:
  {
    "id": "114709",           // 비노출, 이미지 파일 매핑 + DOM key 용도
    "title": "Toy Story",
    "domain": "movie",
    "year": 1995,             // 결측 시 null (프론트에서 "xxxx"로 표시)
    "image": "images/movie/114709.jpg",
    "url": "http://www.imdb.com/title/tt0114709"
  }

보드 데이터 계약:
  {
    "id": "b1",
    "title": "비 오는 날 필요한 조합",   // = 사용자의 검색어
    "section": "recommended" | "mine" | "popular",
    "items": [ {...}, ... ]
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
STATIC_FILES = ["main.html", "board.html", "style.css", "common.js", "README.md", "SPEC.md"]
DEFAULT_OUTPUT = "mockups/vibe_crate"
DEFAULT_TARGET_WIDTH = 400  # px

MOVIE_CSV = "data/canonical/movie_canonical.csv"
MUSIC_CSV = "data/canonical/music_canonical.csv"
BOOK_CSV  = "data/canonical/book_canonical.csv"

# 데모에 쓸 (도메인, id) 풀. id는 movie=imdbId / music=spotify id / book=asin.
# 보드마다 이 풀을 순환 이동(rotate)해서 재사용 — 실제로는 보드별로 서로 다른
# 추천 결과가 API에서 내려와야 하는 자리.
ITEM_POOL = [
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

# 데모 보드 12개 (제목 = 가상의 검색어). 실제로는 유저별 검색 기록에서 생성됨.
BOARD_TITLES = [
    ("recommended", "비 오는 날 필요한 조합"),
    ("recommended", "출근길엔 이 텐션"),
    ("recommended", "혼자 있고 싶은 밤"),
    ("recommended", "여행 가고 싶은 기분"),
    ("mine", "우울한 밤엔 이런 조합"),
    ("mine", "카페인 없이 잠 안 오는 밤"),
    ("mine", "이별 후 회복 중"),
    ("mine", "아무 생각 없고 싶을 때"),
    ("popular", "이번 주 가장 많이 저장된 조합"),
    ("popular", "20대가 많이 찾은 조합"),
    ("popular", "감성 충만 무드"),
    ("popular", "심야 감성 플레이리스트"),
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
    if not os.path.exists(dst_path):
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

    # 아이템 풀 전체를 한 번만 조회 + 이미지 복사
    pool_items = []
    for domain, file_id in ITEM_POOL:
        item = build_item(domain, file_id, tables)
        item["image"] = copy_resized_image(domain, file_id, args.out, args.width)
        pool_items.append(item)

    # 보드마다 풀을 순환 이동(rotate)해서 서로 다른 배치처럼 보이게 함
    boards = []
    for i, (section, title) in enumerate(BOARD_TITLES):
        shift = i % len(pool_items)
        rotated = pool_items[shift:] + pool_items[:shift]
        boards.append({
            "id": f"b{i + 1}",
            "title": title,
            "section": section,
            "items": rotated,
        })

    boards_js = "var BOARDS = " + json.dumps(boards, ensure_ascii=False, indent=2) + ";\n"
    with open(os.path.join(args.out, "boards.js"), "w", encoding="utf-8") as f:
        f.write(boards_js)

    for name in STATIC_FILES:
        shutil.copy(os.path.join(STATIC_DIR, name), os.path.join(args.out, name))

    missing_year = sum(1 for it in pool_items if it["year"] is None)
    print(f"보드 {len(boards)}개 / 아이템 풀 {len(pool_items)}개 처리 완료 "
          f"(year 결측 {missing_year}개 → 프론트에서 'xxxx' 표시)")
    print(f"저장 → {args.out}/main.html, board.html, boards.js, images/")


if __name__ == "__main__":
    main()
