"""
자연어 쿼리 → 영화/음악/책 통합 top-K 추천 (단일 추론).

도메인을 지정하지 않으면 movie + music + book 전체에서 검색하여
유사도 기준으로 통합 랭킹을 반환한다.
--domain 을 지정하면 해당 도메인만 검색한다.

실행 예:
  python3 scripts/inference.py --query "비 오는 날 혼자 보기 좋은"
  python3 scripts/inference.py --query "새벽 드라이브 감성" --top-k 5
  python3 scripts/inference.py --query "고독과 철학" --domain book
  python3 scripts/inference.py --query "재즈바 분위기" --rebuild-index
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MultiModalDataset, collate_fn
from src.data.preprocessing import DOMAIN_CONFIG, prepare_domain_df
from src.models.recommender import DualEncoderModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALL_DOMAINS = ["movie", "music", "book"]
INDEX_CACHE_DIR = Path("data/index_cache")


# ---------------------------------------------------------------------------
# 인덱스 관리
# ---------------------------------------------------------------------------

def index_cache_path(domain: str) -> Path:
    return INDEX_CACHE_DIR / f"{domain}.pt"


@torch.no_grad()
def build_index(
    model: DualEncoderModel,
    domain: str,
    image_dir: str,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, pd.DataFrame]:
    cfg = DOMAIN_CONFIG[domain]
    raw_df = pd.read_csv(cfg["csv"], low_memory=False)
    std_df = prepare_domain_df(domain, raw_df, image_base_dir=image_dir)

    dataset = MultiModalDataset(
        content_texts=std_df["content_text"].tolist(),
        image_paths=std_df["image_path"].tolist(),
        queries=std_df["query"].fillna("").tolist(),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )

    model.eval()
    all_embeddings = []
    for batch in tqdm(loader, desc=f"Encoding {domain}"):
        z, _, _ = model.encode_content(batch["content_text"], batch["content_image"])
        all_embeddings.append(z.cpu())

    z_contents_n = F.normalize(torch.cat(all_embeddings), p=2, dim=1)
    meta_df = _build_meta_df(domain, raw_df, std_df)
    return z_contents_n, meta_df


def save_index(domain: str, z_contents_n: torch.Tensor, meta_df: pd.DataFrame) -> None:
    INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"z_contents_n": z_contents_n, "meta": meta_df.to_dict("records")},
        index_cache_path(domain),
    )
    logger.info(f"Index saved → {index_cache_path(domain)}")


def load_index(domain: str) -> tuple[torch.Tensor, pd.DataFrame]:
    data = torch.load(index_cache_path(domain), map_location="cpu")
    return data["z_contents_n"], pd.DataFrame(data["meta"])


def _build_meta_df(domain: str, raw_df: pd.DataFrame, std_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for i, (_, row) in enumerate(raw_df.iterrows()):
        item_id = std_df.iloc[i]["item_id"]
        if domain == "movie":
            title = str(row.get("Title", ""))
            extra = {"genre": str(row.get("Genre", "")), "poster": str(row.get("Poster", ""))}
        elif domain == "music":
            title = str(row.get("name", ""))
            extra = {"artists": str(row.get("artists", "")),
                     "album": str(row.get("album_name", "")),
                     "genre": str(row.get("genre", "")),
                     "img": str(row.get("img", ""))}
        else:  # book
            title = str(row.get("title", ""))
            extra = {"author": str(row.get("author", "")),
                     "category": str(row.get("category_name", "")),
                     "image": str(row.get("imgUrl", ""))}
        records.append({"item_id": item_id, "domain": domain, "title": title, "extra": extra})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 통합 검색
# ---------------------------------------------------------------------------

@torch.no_grad()
def search(
    model: DualEncoderModel,
    query: str,
    indexes: dict[str, tuple[torch.Tensor, pd.DataFrame]],
    top_k: int,
) -> list[dict]:
    """
    전체(또는 지정) 도메인 인덱스를 하나로 합쳐 top-K 검색.
    도메인 간 임베딩 공간이 동일하므로 점수를 직접 비교할 수 있다.
    """
    model.eval()
    z_query = model.encode_query([query])                # (1, 768)
    z_query_n = F.normalize(z_query, p=2, dim=1).cpu()  # (1, 768)

    all_scores, all_meta = [], []
    for domain, (z_contents_n, meta_df) in indexes.items():
        scores = (z_query_n @ z_contents_n.T).squeeze(0)  # (N_domain,)
        all_scores.append(scores)
        all_meta.append(meta_df)

    merged_scores = torch.cat(all_scores)           # (N_total,)
    merged_meta   = pd.concat(all_meta, ignore_index=True)

    top_idx = merged_scores.topk(top_k).indices.tolist()

    results = []
    for idx in top_idx:
        row = merged_meta.iloc[idx]
        results.append({
            "item_id": str(row["item_id"]),
            "domain":  row["domain"],
            "score":   float(merged_scores[idx]),
            "title":   row["title"],
            **row["extra"],
        })
    return results


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------

DOMAIN_EMOJI = {"movie": "🎬", "music": "🎵", "book": "📚"}

def print_results(query: str, domains: list[str], results: list[dict]) -> None:
    domain_str = " + ".join(d.upper() for d in domains)
    print(f'\n{"=" * 62}')
    print(f'  Query  : "{query}"')
    print(f"  Search : {domain_str}")
    print(f'{"=" * 62}')
    for rank, item in enumerate(results, 1):
        icon = DOMAIN_EMOJI.get(item["domain"], "")
        print(f"  {rank:>2}. [{item['score']:.4f}] {icon} {item['title']}")
        extras = {k: v for k, v in item.items()
                  if k not in ("item_id", "domain", "score", "title", "poster", "img", "image")
                  and v and str(v) not in ("nan", "")}
        if extras:
            print(f"      {' | '.join(f'{k}: {v}' for k, v in extras.items())}")
    print(f'{"=" * 62}\n')


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 모델 로드
    logger.info(f"Loading model from {args.model_path}")
    model = DualEncoderModel()
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # 검색 대상 도메인
    target_domains = [args.domain] if args.domain else ALL_DOMAINS

    # 인덱스 로드 또는 생성
    indexes: dict[str, tuple[torch.Tensor, pd.DataFrame]] = {}
    for domain in target_domains:
        cache = index_cache_path(domain)
        if not args.rebuild_index and cache.exists():
            logger.info(f"[{domain}] Loading cached index from {cache}")
            indexes[domain] = load_index(domain)
        else:
            logger.info(f"[{domain}] Building index...")
            z, meta = build_index(model, domain, args.image_dir, args.batch_size, args.num_workers)
            save_index(domain, z, meta)
            indexes[domain] = (z, meta)

    total = sum(len(meta) for _, meta in indexes.values())
    logger.info(f"Index ready: {total:,} items across {list(indexes.keys())}")

    # 검색 및 출력
    results = search(model, args.query, indexes, args.top_k)
    print_results(args.query, target_domains, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-domain query-based recommendation")
    parser.add_argument("--query",         required=True,  help="자연어 검색 쿼리")
    parser.add_argument("--domain",        default=None,   choices=["movie", "music", "book"],
                        help="특정 도메인만 검색 (기본: 전체 도메인 통합 검색)")
    parser.add_argument("--model-path",    default="models/trained_model.pt")
    parser.add_argument("--image-dir",     default="data/images")
    parser.add_argument("--top-k",         type=int, default=10)
    parser.add_argument("--batch-size",    type=int, default=32)
    parser.add_argument("--num-workers",   type=int, default=4)
    parser.add_argument("--rebuild-index", action="store_true", help="캐시 무시하고 인덱스 재생성")
    parser.add_argument("--device",        default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    main(args)
