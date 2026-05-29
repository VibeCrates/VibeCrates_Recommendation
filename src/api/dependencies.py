"""
Dependency injection for FastAPI — 모델 로드 및 아이템 인덱스 관리.
"""
import logging
import os
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.preprocessing import DOMAIN_CONFIG, prepare_domain_df
from src.models.recommender import DualEncoderModel
from .schemas import RecommendationItem

logger = logging.getLogger(__name__)

MODEL_PATH  = os.getenv("MODEL_PATH",  "models/trained_model.pt")
IMAGE_DIR   = os.getenv("IMAGE_DIR",   "data/images")
DEVICE      = os.getenv("DEVICE",      "cuda" if torch.cuda.is_available() else "cpu")


class ModelManager:
    """
    모델과 도메인별 아이템 임베딩 인덱스를 관리.

    - model: DualEncoderModel (추론 전용)
    - indexes[domain]: (z_contents_n, meta_df) 쌍
      z_contents_n : (N, 768) L2-정규화된 아이템 임베딩 CPU tensor
      meta_df      : 아이템 메타데이터 DataFrame (item_id, title, extra)
    """

    def __init__(self):
        self.model: Optional[DualEncoderModel] = None
        self.device = torch.device(DEVICE)
        self.indexes: dict[str, tuple[torch.Tensor, pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    # 모델
    # ------------------------------------------------------------------

    def load_model(self, model_path: str = MODEL_PATH) -> None:
        logger.info(f"Loading model from {model_path}")
        model = DualEncoderModel()
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model
        logger.info("Model loaded.")

    def is_model_ready(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    # 아이템 인덱스
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_index(self, domain: str, batch_size: int = 64) -> None:
        """
        도메인 CSV를 읽어 전체 아이템 임베딩을 생성하고 메모리에 캐싱.
        모델이 로드된 후 호출해야 함.
        """
        if not self.is_model_ready():
            raise RuntimeError("모델을 먼저 로드하세요.")

        cfg = DOMAIN_CONFIG[domain]
        df = pd.read_csv(cfg["csv"], low_memory=False)
        std_df = prepare_domain_df(domain, df, image_base_dir=IMAGE_DIR)

        logger.info(f"[{domain}] Building index for {len(std_df):,} items...")

        from torch.utils.data import DataLoader as TorchDataLoader
        from src.data.dataset import MultiModalDataset, collate_fn

        dataset = MultiModalDataset(
            content_texts=std_df["content_text"].tolist(),
            image_paths=std_df["image_path"].tolist(),
            queries=std_df["query"].fillna("").tolist(),
        )
        loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        all_embeddings = []
        self.model.eval()
        for batch in loader:
            z, _, _ = self.model.encode_content(batch["content_text"], batch["content_image"])
            all_embeddings.append(z.cpu())

        z_contents = torch.cat(all_embeddings)
        z_contents_n = F.normalize(z_contents, p=2, dim=1)

        # 메타 정보 구성 (도메인별)
        meta_df = _build_meta_df(domain, df, std_df)

        self.indexes[domain] = (z_contents_n, meta_df)
        logger.info(f"[{domain}] Index built: {z_contents_n.shape}")

    # ------------------------------------------------------------------
    # 검색
    # ------------------------------------------------------------------

    @torch.no_grad()
    def search(self, query: str, top_k: int, domain: str | None = None) -> list[RecommendationItem]:
        """
        domain=None 이면 준비된 모든 도메인을 합쳐 통합 검색.
        domain 지정 시 해당 도메인만 검색.
        """
        target = {domain: self.indexes[domain]} if domain else self.indexes

        self.model.eval()
        z_query = self.model.encode_query([query])
        z_query_n = F.normalize(z_query, p=2, dim=1).cpu()

        all_scores, all_meta = [], []
        for d, (z_contents_n, meta_df) in target.items():
            scores = (z_query_n @ z_contents_n.T).squeeze(0)
            all_scores.append(scores)
            all_meta.append(meta_df)

        merged_scores = torch.cat(all_scores)
        merged_meta   = pd.concat(all_meta, ignore_index=True)

        top_idx = merged_scores.topk(top_k).indices.tolist()
        results = []
        for idx in top_idx:
            row = merged_meta.iloc[idx]
            results.append(RecommendationItem(
                item_id=str(row["item_id"]),
                domain=str(row["domain"]),
                score=float(merged_scores[idx]),
                title=str(row["title"]),
                extra=row.get("extra"),
            ))
        return results

    # ------------------------------------------------------------------
    # 아이템 메타 조회
    # ------------------------------------------------------------------

    def get_item_info(self, domain: str, item_id: str) -> Optional[dict]:
        if domain not in self.indexes:
            return None
        _, meta_df = self.indexes[domain]
        rows = meta_df[meta_df["item_id"].astype(str) == item_id]
        if rows.empty:
            return None
        row = rows.iloc[0]
        return {"title": row["title"], **(row.get("extra") or {})}


# ---------------------------------------------------------------------------
# 도메인별 메타 DataFrame 생성
# ---------------------------------------------------------------------------

def _build_meta_df(domain: str, raw_df: pd.DataFrame, std_df: pd.DataFrame) -> pd.DataFrame:
    """item_id, title, extra(dict) 컬럼을 가진 DataFrame."""
    records = []
    for i, (_, raw_row) in enumerate(raw_df.iterrows()):
        item_id = std_df.iloc[i]["item_id"]
        if domain == "movie":
            title = str(raw_row.get("Title", ""))
            extra = {"genre": str(raw_row.get("Genre", "")), "poster": str(raw_row.get("Poster", ""))}
        elif domain == "music":
            title = str(raw_row.get("name", ""))
            extra = {"artists": str(raw_row.get("artists", "")), "album": str(raw_row.get("album_name", "")),
                     "genre": str(raw_row.get("genre", "")), "img": str(raw_row.get("img", ""))}
        else:  # book
            title = str(raw_row.get("title", ""))
            extra = {"author": str(raw_row.get("author", "")),
                     "category": str(raw_row.get("category_name", "")),
                     "image": str(raw_row.get("imgUrl", ""))}
        records.append({"item_id": item_id, "title": title, "extra": extra})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# FastAPI 의존성
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_model_manager() -> ModelManager:
    return _manager


_manager = ModelManager()


async def initialize_dependencies() -> None:
    logger.info("Initializing dependencies...")
    try:
        _manager.load_model()
        for domain in ("movie", "music", "book"):
            try:
                _manager.build_index(domain)
            except Exception as e:
                logger.warning(f"[{domain}] Index build failed: {e}")
    except Exception as e:
        logger.error(f"Model load failed: {e}")


async def cleanup_dependencies() -> None:
    logger.info("Cleaning up...")
    _manager.indexes.clear()
    _manager.model = None
