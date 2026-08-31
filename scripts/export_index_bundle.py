#!/usr/bin/env python
"""백엔드 벡터 DB에 넣을 번들을 만든다 — 아이템 벡터 + 평면 메타 + manifest.

왜 필요한가 (2026-08-24 결정 D9=B) — 검색은 백엔드가 자기 벡터 DB에서 한다. 추천 서버는
검색어를 벡터로 바꿔 줄 뿐이므로, 검색 대상인 **아이템 벡터를 백엔드가 들고 있어야** 한다.
그런데 벡터만 넘기면 화면을 그릴 수 없다. 프론트 계약(`scripts/vibe_crate/SPEC.md`)의
아이템은 `{id, title, domain, year, image, url}` 평면 구조이므로 그 여섯 필드를 같은
순서로 함께 넘긴다.

이 스크립트가 지키는 두 가지 불변식:

  1. **행 순서** — `{domain}_vectors.npy`의 i번째 행과 `{domain}_items.parquet`의 i번째
     행은 같은 아이템이다. 인덱스(`indexes/{domain}_meta.parquet`)의 순서를 그대로 따르며,
     canonical CSV는 id로 조인해 붙이기만 한다. CSV를 다시 읽어 순서를 만들면 어긋난다.
  2. **출처 표시** — manifest의 model_version은 API 응답의 것과 같은 함수에서 나온다
     (`src.api.dependencies.checkpoint_version`). 이 값이 다르면 쿼리 벡터와 아이템 벡터가
     다른 체크포인트에서 나온 것이고, 그때 검색은 오류 없이 성공하고 결과만 엉뚱해진다.

image 필드는 **파일이 실제로 있는 것만** 채운다(`--image-list-dir`). 없는 것을 경로로
채워 두면 백엔드가 404를 받고서야 알게 되고, 콜라주는 빈 칸으로 그려진다. music은 커버가
66.8%뿐이라 이 구분이 특히 중요하다.

사용:
  python scripts/export_index_bundle.py --out dist/bundle_20260824 \\
      --image-list-dir /tmp/imagelists
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api.dependencies import checkpoint_version   # noqa: E402

DOMAINS = ("movie", "music", "book")

# 도메인별 id 형태. canonical CSV에 따옴표·줄바꿈이 깨져 컬럼이 밀린 행이 섞여 있고
# (movie 1건 실측, 2026-08-31), 그런 행은 id 자리에 줄거리 문장이 들어앉는다. 제목도
# 본문도 없으니 벡터도 의미가 없다. 넘기면 백엔드 DB에 쓰레기 키가 한 건 박힌다.
ID_PATTERNS = {
    "movie": r"\d+",                  # imdbId
    "music": r"[0-9A-Za-z]+",         # Spotify 트랙 id (base62)
    "book":  r"(kdl_|gr_|bx_).+",     # 출처 접두사 + 원본 키
}

# 계약의 여섯 필드. 순서까지 계약대로 맞춘다.
CONTRACT_COLUMNS = ["id", "title", "domain", "year", "image", "url"]

# 계약에는 없지만 번들에만 넣는 컬럼. vectors.npy의 몇 번째 행인지를 명시한다.
#   왜: 벡터 파일에는 id가 없고, 벡터와 아이템은 행 순서로만 이어져 있다. 그 순서가
#   파일 어디에도 적혀 있지 않으면 암묵적 약속으로만 남는다. 특히 Qdrant(정수·UUID만
#   허용)나 FAISS(int64만 허용)처럼 문자열 키를 못 쓰는 DB에서는 이 행 번호가 곧 DB의
#   기본키가 되므로, parquet을 재정렬하는 순간 매칭이 통째로 어긋난다.
ROW_COLUMN = "row"

# 벡터 DB의 기본키로 쓸 전역 고유 정수. row는 도메인 안에서만 0부터 매겨지므로
# 세 도메인을 한 컬렉션에 넣으면 movie의 0번과 music의 0번이 충돌해 서로 덮어쓴다
# (실측: movie 39,515개가 music·book과 전부 겹친다). 도메인 코드를 앞에 붙여 가른다.
#   movie 1_000_000_000+row / music 2_000_000_000+row / book 3_000_000_000+row
# Qdrant의 point id는 부호 없는 64비트 정수라 이 범위를 넉넉히 담는다.
POINT_ID_COLUMN = "point_id"
DOMAIN_CODE = {"movie": 1, "music": 2, "book": 3}
DOMAIN_STRIDE = 1_000_000_000


def _year(value) -> int | None:
    """'1995-11-22' / '1995' / 1995.0 → 1995. 못 읽으면 None(계약: 결측은 null)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if len(text) >= 4 and text[:4].isdigit():
        year = int(text[:4])
        if 1000 <= year <= 2100:
            return year
    return None


def _clean_movie_title(title: str) -> tuple[str, int | None]:
    """'Toy Story (1995)' → ('Toy Story', 1995).

    movie 메타의 제목에는 연도가 괄호로 붙어 있는데 계약은 title과 year를 나눠 받는다.
    release_date가 1.3% 비어 있으므로 여기서 뽑은 연도를 폴백으로 쓴다.
    """
    text = str(title).strip()
    if text.endswith(")") and "(" in text:
        head, _, tail = text.rpartition("(")
        year = _year(tail[:-1])
        if year:
            return head.strip(), year
    return text, None


def load_canonical(domain: str, canonical_dir: str) -> pd.DataFrame:
    """id → 계약 필드의 원천이 되는 컬럼만 뽑아 온다."""
    if domain == "movie":
        df = pd.read_csv(os.path.join(canonical_dir, "movie_canonical.csv"),
                         usecols=["imdbId", "release_date", "Imdb Link"], low_memory=False)
        return df.rename(columns={"imdbId": "id"}).assign(id=lambda d: d["id"].astype(str))
    if domain == "music":
        df = pd.read_csv(os.path.join(canonical_dir, "music_canonical.csv"),
                         usecols=["id", "year"], low_memory=False)
        return df.assign(id=lambda d: d["id"].astype(str))
    df = pd.read_csv(os.path.join(canonical_dir, "book_canonical_v2.csv"),
                     usecols=["asin", "publishedDate", "isbn", "source"], low_memory=False)
    return df.rename(columns={"asin": "id"}).assign(id=lambda d: d["id"].astype(str))


def build_url(domain: str, row: pd.Series) -> str | None:
    """상세 페이지 링크. 계약의 `url`은 hover 클릭용이며, 표지 이미지와는 다른 필드다."""
    if domain == "movie":
        link = row.get("Imdb Link")
        return str(link) if isinstance(link, str) and link.strip() else None
    if domain == "music":
        # 주의: music CSV의 preview 컬럼은 30초 미리듣기 오디오라 계약의 url이 아니다.
        return f"https://open.spotify.com/track/{row['id']}"
    # book — v2에는 상품 링크 컬럼이 아예 없다(출처 82%가 Amazon 상품이 아니다).
    # isbn이 있으면 Open Library, kindle이면 접두사를 뗀 ASIN으로 Amazon.
    isbn = row.get("isbn")
    if isinstance(isbn, str) and isbn.strip():
        return f"https://openlibrary.org/isbn/{isbn.strip()}"
    if str(row["id"]).startswith("kdl_"):
        return f"https://www.amazon.com/dp/{str(row['id'])[4:]}"
    return None


def build_items(domain: str, meta: pd.DataFrame, canonical: pd.DataFrame,
                images: set[str] | None) -> pd.DataFrame:
    meta = meta.copy()
    meta["id"] = meta["item_id"].astype(str)

    titles, title_years = [], []
    for t in meta["title"]:
        if domain == "movie":
            title, year = _clean_movie_title(t)
        else:
            title, year = str(t).strip(), None
        titles.append(title)
        title_years.append(year)
    meta["title"] = titles
    meta["_title_year"] = title_years

    # canonical에 같은 id가 여러 행인 경우가 있다(백로그 A6: movie 593행 / music 354행).
    # 메타를 붙이는 목적에서는 같은 아이템의 사본이므로 첫 행만 쓴다. 실제 중복 제거는
    # 아래 dedupe 단계에서 벡터와 함께 한다.
    canonical = canonical.drop_duplicates(subset="id", keep="first")
    merged = meta.merge(canonical, on="id", how="left", validate="m:1")

    year_source = {"movie": "release_date", "music": "year", "book": "publishedDate"}[domain]
    years = [_year(v) for v in merged[year_source]]
    if domain == "movie":  # release_date 결측은 제목 속 연도로 메운다
        years = [y if y is not None else ty for y, ty in zip(years, merged["_title_year"])]
    # Int64(nullable)로 둔다. 기본 float가 되면 계약의 1995가 1995.0으로 나가고,
    # 결측은 NaN이 되어 JSON에서 null이 아니라 NaN으로 직렬화된다.
    merged["year"] = pd.array(years, dtype="Int64")

    merged["url"] = [build_url(domain, row) for _, row in merged.iterrows()]

    # 파일이 있는 것만 경로를 준다. 나머지는 null — 백엔드가 콜라주에서 뺄 수 있어야 한다.
    if images is None:
        merged["image"] = [f"images/{domain}/{i}.jpg" for i in merged["id"]]
    else:
        merged["image"] = [f"images/{domain}/{i}.jpg" if f"{i}.jpg" in images else None
                           for i in merged["id"]]

    merged["domain"] = domain
    return merged[CONTRACT_COLUMNS]  # row는 필터링이 모두 끝난 뒤에 붙인다


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", default="indexes")
    ap.add_argument("--canonical-dir", default="data/canonical")
    ap.add_argument("--model-path", default=os.getenv("MODEL_PATH", "models/trained_model.pt"))
    ap.add_argument("--image-list-dir", default=None,
                    help="{domain}.txt에 사용 가능한 이미지 파일명 목록. 없으면 전부 경로를 채운다")
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS))
    ap.add_argument("--no-dedupe-ids", dest="dedupe_ids", action="store_false",
                    help="중복 id를 그대로 둔다(기본은 제거)")
    ap.add_argument("--min-join-rate", type=float, default=0.99,
                    help="canonical 조인 성공률 하한. 미만이면 중단한다")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    version = checkpoint_version(args.model_path)
    manifest = {
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dim": None,
        "metric": "inner_product",
        "normalized": True,
        "note": "items의 row 컬럼이 vectors.npy의 행 번호다(vectors[row] ↔ 그 행). "
                "문자열 키를 못 쓰는 벡터 DB(Qdrant 등)에서는 point_id를 기본키로, "
                "id는 payload에 넣는다. row는 도메인 안에서만 고유하므로 한 컬렉션에 "
                "세 도메인을 넣을 때 기본키로 쓰면 충돌한다. "
                "쿼리 벡터는 POST /api/v1/search/vector 로 받고, 응답의 model_version이 "
                "이 값과 다르면 검색 결과가 무의미하다.",
        "domains": {},
    }
    if version == "unknown":
        print(f"! 체크포인트를 읽을 수 없다: {args.model_path} — manifest에 unknown이 박힌다")

    for domain in args.domains:
        vectors = torch.load(os.path.join(args.index_dir, f"{domain}_embeddings.pt"),
                             map_location="cpu")
        meta = pd.read_parquet(os.path.join(args.index_dir, f"{domain}_meta.parquet"))
        if len(meta) != vectors.shape[0]:
            raise SystemExit(f"[{domain}] 행 수 불일치: 벡터 {vectors.shape[0]} / 메타 {len(meta)}")

        norms = vectors.norm(dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
            raise SystemExit(f"[{domain}] L2 정규화 상태가 아니다 — 내적을 코사인으로 쓸 수 없다")

        images = None
        if args.image_list_dir:
            list_path = os.path.join(args.image_list_dir, f"{domain}.txt")
            with open(list_path) as f:
                images = {line.strip() for line in f if line.strip()}

        canonical = load_canonical(domain, args.canonical_dir)
        items = build_items(domain, meta, canonical, images)

        # 인덱스 자체에 같은 아이템이 두 행으로 들어 있다(백로그 A6). 그대로 넘기면
        # 백엔드 검색 결과에 같은 것이 두 번 뜬다 — 화면에서 바로 보이는 종류의 문제다.
        # 행 순서 불변식을 지키기 위해 items와 vectors를 **같은 마스크로** 함께 거른다.
        # 형식이 깨진 id를 벡터와 함께 걸러낸다. dedupe와 같은 이유로 마스크를 함께 쓴다.
        valid = items["id"].str.fullmatch(ID_PATTERNS[domain]).fillna(False)
        if not valid.all():
            print(f"[{domain}] id 형식이 깨진 행 {int((~valid).sum())}건 제거 "
                  f"(예: {items.loc[~valid, 'id'].iloc[0][:50]!r})")
            vectors = vectors[torch.as_tensor(valid.to_numpy().copy())]
            items = items[valid].reset_index(drop=True)

        if args.dedupe_ids:
            keep = ~items["id"].duplicated(keep="first")
            dropped = int((~keep).sum())
            if dropped:
                items = items[keep].reset_index(drop=True)
                vectors = vectors[torch.as_tensor(keep.to_numpy().copy())]
                print(f"[{domain}] 중복 id {dropped}건 제거 (백로그 A6)")

        # A7과 같은 취지의 가드 — 조인이 깨진 채로 넘어가면 year·url이 조용히 전부 비고,
        # 백엔드는 "우리 쪽 문제인가" 부터 의심하게 된다.
        join_rate = items["id"].isin(canonical["id"]).mean()
        if join_rate < args.min_join_rate:
            raise SystemExit(f"[{domain}] canonical 조인 성공률 {join_rate:.1%} — "
                             f"하한 {args.min_join_rate:.0%} 미만이라 중단한다")

        # 제거가 모두 끝난 뒤에 번호를 매긴다. 거르기 전에 매기면 중간에 구멍이 뚫려
        # vectors.npy의 실제 행 번호와 어긋난다.
        items = items.reset_index(drop=True)
        items[ROW_COLUMN] = np.arange(len(items), dtype=np.int64)
        items[POINT_ID_COLUMN] = (DOMAIN_CODE[domain] * DOMAIN_STRIDE
                                  + items[ROW_COLUMN]).astype(np.int64)

        vec_path  = os.path.join(args.out, f"{domain}_vectors.npy")
        item_path = os.path.join(args.out, f"{domain}_items.parquet")
        np.save(vec_path, vectors.numpy().astype(np.float32))
        items.to_parquet(item_path, index=False)

        manifest["dim"] = int(vectors.shape[1])
        manifest["domains"][domain] = {
            "count": int(len(items)),
            "vectors_file": os.path.basename(vec_path),
            "items_file": os.path.basename(item_path),
            "vectors_sha256": sha256(vec_path),
            "items_sha256": sha256(item_path),
            "coverage": {
                "year":  round(float(items["year"].notna().mean()), 4),
                "image": round(float(items["image"].notna().mean()), 4),
                "url":   round(float(items["url"].notna().mean()), 4),
            },
        }
        cov = manifest["domains"][domain]["coverage"]
        print(f"[{domain}] {len(items):>7,}건  조인 {join_rate:.1%}  "
              f"year {cov['year']:.1%} / image {cov['image']:.1%} / url {cov['url']:.1%}")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n번들: {args.out}  (model_version={version})")


if __name__ == "__main__":
    main()
