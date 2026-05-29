"""
DualEncoderModel retrieval 성능 평가.

동작 방식:
  1. 모델 로드 (torch state_dict)
  2. 도메인 CSV → prepare_domain_df → MultiModalDataset
  3. 전체 아이템 encode_content → z_content 인덱스
  4. 전체 아이템 encode_query  → z_query
  5. 각 query에 대해 정답 아이템의 cosine 유사도 순위 계산 (청크 처리)
  6. Recall@K / NDCG@K / MRR 출력

실행 예:
  python3 scripts/evaluate.py --domain movie --model-path models/trained_model.pt
  python3 scripts/evaluate.py --domain music --top-k 1 5 10 20 --batch-size 64
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
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


# ---------------------------------------------------------------------------
# 메트릭
# ---------------------------------------------------------------------------

def recall_at_k(ranks: np.ndarray, k: int) -> float:
    """rank < k인 비율 (relevant item이 top-K에 포함된 비율)."""
    return float((ranks < k).mean())


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    """NDCG@K. relevant item이 1개이므로 IDCG = 1, DCG = 1/log2(rank+2)."""
    hits = ranks < k
    dcg = np.where(hits, 1.0 / np.log2(ranks + 2.0), 0.0)
    return float(dcg.mean())


def mean_reciprocal_rank(ranks: np.ndarray) -> float:
    return float((1.0 / (ranks + 1.0)).mean())


# ---------------------------------------------------------------------------
# 임베딩 추출
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_all(
    model: DualEncoderModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    전체 데이터셋을 순회해 z_content와 z_query를 추출.

    Returns:
        z_contents: (N, D) CPU tensor
        z_queries:  (N, D) CPU tensor
    """
    model.eval()
    all_content, all_query = [], []

    for batch in tqdm(loader, desc="Encoding"):
        z_content, _, _ = model.encode_content(batch["content_text"], batch["content_image"])
        z_query = model.encode_query(batch["query"])
        all_content.append(z_content.cpu())
        all_query.append(z_query.cpu())

    return torch.cat(all_content), torch.cat(all_query)


# ---------------------------------------------------------------------------
# 순위 계산
# ---------------------------------------------------------------------------

def compute_ranks(
    z_queries_n: torch.Tensor,
    z_contents_n: torch.Tensor,
    chunk_size: int = 512,
) -> np.ndarray:
    """
    각 query i에 대해 정답 아이템 i의 순위(0-indexed)를 계산.
    (N, N) 전체 행렬을 한 번에 만들지 않고 청크 단위로 처리해 OOM 방지.

    rank=0 → 1위 (가장 유사), rank=k → k+1위

    self_sim을 chunk_sim 대각선에서 추출해 동일 연산 경로를 보장,
    float32 matmul vs element-wise 오차로 인한 순위 오류 방지.
    """
    N = z_queries_n.shape[0]
    ranks = np.empty(N, dtype=np.int64)

    for start in tqdm(range(0, N, chunk_size), desc="Ranking"):
        end = min(start + chunk_size, N)
        chunk_sim = z_queries_n[start:end] @ z_contents_n.T  # (chunk, N)
        local_idx = torch.arange(end - start)
        global_idx = torch.arange(start, end)
        self_sim = chunk_sim[local_idx, global_idx].unsqueeze(1)  # (chunk, 1)
        ranks[start:end] = (chunk_sim > self_sim).sum(dim=1).numpy()

    return ranks


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

    # 데이터 로드
    csv_path = args.data_path or DOMAIN_CONFIG[args.domain]["csv"]
    logger.info(f"Loading {args.domain} data from {csv_path}")
    raw_df = pd.read_csv(csv_path, low_memory=False)

    if "query" not in raw_df.columns or raw_df["query"].isna().all():
        logger.error("query 컬럼이 없거나 비어 있습니다. generate_queries.py를 먼저 실행하세요.")
        sys.exit(1)

    raw_df = raw_df[
        raw_df["query"].notna() & (raw_df["query"].astype(str).str.strip() != "")
    ].reset_index(drop=True)
    logger.info(f"평가 대상: {len(raw_df):,}개")

    std_df = prepare_domain_df(args.domain, raw_df, image_base_dir=args.image_dir)

    dataset = MultiModalDataset(
        content_texts=std_df["content_text"].tolist(),
        image_paths=std_df["image_path"].tolist(),
        queries=std_df["query"].tolist(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # 임베딩 추출
    z_contents, z_queries = encode_all(model, loader, device)
    z_contents_n = F.normalize(z_contents, p=2, dim=1)
    z_queries_n  = F.normalize(z_queries,  p=2, dim=1)

    # 순위 계산
    ranks = compute_ranks(z_queries_n, z_contents_n, chunk_size=args.chunk_size)

    # 결과 출력
    ks = sorted(set(args.top_k))
    print("\n" + "=" * 55)
    print(f"  Evaluation — {args.domain.upper()}  ({len(ranks):,} samples)")
    print("=" * 55)
    print(f"  {'Metric':<22} {'Value':>10}")
    print("-" * 55)
    print(f"  {'MRR':<22} {mean_reciprocal_rank(ranks):>10.4f}")
    for k in ks:
        print(f"  {'Recall@'+str(k):<22} {recall_at_k(ranks, k):>10.4f}")
        print(f"  {'NDCG@'+str(k):<22} {ndcg_at_k(ranks, k):>10.4f}")
    print("-" * 55)
    print(f"  {'Median rank':<22} {int(np.median(ranks)) + 1:>10,}")
    print(f"  {'Mean rank':<22} {ranks.mean() + 1:>10.1f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DualEncoderModel retrieval performance")
    parser.add_argument("--domain",      required=True, choices=["movie", "music", "book"])
    parser.add_argument("--model-path",  default="models/trained_model.pt")
    parser.add_argument("--data-path",   default=None, help="CSV 경로 (기본: 도메인 기본 CSV)")
    parser.add_argument("--image-dir",   default="data/images")
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--chunk-size",  type=int, default=512,  help="순위 계산 청크 크기")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-k",       type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--device",      default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    main(args)
