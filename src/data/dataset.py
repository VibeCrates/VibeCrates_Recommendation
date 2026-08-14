"""
Custom PyTorch Dataset for multi-modal recommendation data.
Handles text, image, and query loading.
"""
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Dict, Optional
import numpy as np


class MultiModalDataset(Dataset):
    """
    A PyTorch Dataset for multi-modal recommendation data.

    Each sample contains:
    - content_text: Product/content description (string)
    - content_image: Pre-computed CLIP embedding (Tensor) if image_embeddings provided,
                     otherwise a PIL Image loaded from disk.
    - query: List[str] parsed from DSV (e.g. "q1|q2|q3" → ["q1","q2","q3"])
    """

    def __init__(
        self,
        content_texts: List[str],
        image_paths: List[str],
        queries: List[str],
        image_size: Tuple[int, int] = (224, 224),
        image_embeddings: Optional[Dict[str, np.ndarray]] = None,
        sample_one_query: bool = False,
    ):
        assert len(content_texts) == len(image_paths) == len(queries), \
            "All input lists must have the same length"

        self.content_texts = content_texts
        self.image_paths = image_paths
        self.queries = queries
        self.image_size = image_size
        self.image_embeddings = image_embeddings  # {image_path: np.array(D)}
        # 개선안 2 (설계 노트 근본원인 2) — 학습 시 아이템당 쿼리 1개만 무작위로 쓴다.
        # encode_query가 Poet/Space/Philosopher 3개를 mean-pooling 하면 콘텐츠는 세 방향의
        # 타협 centroid에 정렬되고, 개별 poet 쿼리는 한 번도 검색 앵커로 학습되지 않는다.
        # 추론 시 순수 poet 쿼리가 그 centroid에서 벗어나 실패하는 것이 6월 평가에서 poet이
        # 전역 최하였던 원인 중 하나다. 매 스텝 하나를 뽑으면 에폭을 거치며 콘텐츠가 세
        # 페르소나를 각각 직접 앵커로 경험하고, 같은 아이템의 다른 페르소나가 배치 안에서
        # false negative가 되는 일도 없다(설계 노트의 "방법 B").
        # mean-pooling은 추론 경로(멀티 쿼리 입력)에만 남는다 — 이 플래그는 학습에서만 켠다.
        self.sample_one_query = sample_one_query
        # 임베딩을 쓰는 경우 배치 전체가 텐서여야 한다 (collate_fn은 첫 항목 타입만 보고
        # torch.stack 한다 — 하나라도 PIL이 섞이면 거기서 터진다).
        self.embedding_dim = (
            len(next(iter(image_embeddings.values()))) if image_embeddings else None
        )
        self.missing_embeddings = 0

    def __len__(self) -> int:
        return len(self.content_texts)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        content_text = self.content_texts[idx]
        image_path = self.image_paths[idx]

        if self.image_embeddings is not None:
            emb = self.image_embeddings.get(image_path)
            if emb is not None:
                image = torch.tensor(emb, dtype=torch.float32)
            else:
                # 임베딩이 없는 항목(로컬 파일 없음 → image_path가 http URL이거나 빈 문자열).
                # 여기서 아래 PIL 경로로 떨어뜨리면 세 가지가 한꺼번에 잘못된다:
                #   1) 배치에 텐서와 PIL이 섞여 collate_fn의 torch.stack이 터진다,
                #   2) 학습 스텝마다 원격 이미지를 받으러 나간다(에폭마다 반복),
                #   3) 실패 시 전부 같은 검은 이미지가 되어 "이미지 없음" 항목끼리
                #      CLIP 공간의 한 점에 뭉친다 — 조용히 변별력을 깎는다.
                # 0 벡터로 통일해 "시각 신호 없음"을 명시한다.
                image = torch.zeros(self.embedding_dim, dtype=torch.float32)
                self.missing_embeddings += 1
        else:
            try:
                if str(image_path).startswith("http"):
                    import requests
                    from io import BytesIO
                    r = requests.get(image_path, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                    r.raise_for_status()
                    image = Image.open(BytesIO(r.content)).convert("RGB")
                else:
                    image = Image.open(image_path).convert("RGB")
            except Exception:
                image = Image.new("RGB", self.image_size)
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)

        raw = self.queries[idx]
        if isinstance(raw, str) and raw.strip():
            query = [q.strip() for q in raw.split("|") if q.strip()]
        else:
            query = []

        if self.sample_one_query and len(query) > 1:
            query = [random.choice(query)]

        return {
            "content_text": content_text,
            "content_image": image,
            "query": query,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Converts a list of samples into a batch.

    - content_text: List[str]
    - content_image: Tensor (B, D) if pre-computed embeddings, else List[PIL.Image]
    - query: List[List[str]]
    """
    images = [s["content_image"] for s in batch]
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images)  # (B, D)
    return {
        "content_text": [s["content_text"] for s in batch],
        "content_image": images,
        "query": [s["query"] for s in batch],
    }
