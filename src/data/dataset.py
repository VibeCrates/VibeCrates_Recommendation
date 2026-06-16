"""
Custom PyTorch Dataset for multi-modal recommendation data.
Handles text, image, and query loading.
"""
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
    ):
        assert len(content_texts) == len(image_paths) == len(queries), \
            "All input lists must have the same length"

        self.content_texts = content_texts
        self.image_paths = image_paths
        self.queries = queries
        self.image_size = image_size
        self.image_embeddings = image_embeddings  # {image_path: np.array(D)}

    def __len__(self) -> int:
        return len(self.content_texts)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        content_text = self.content_texts[idx]
        image_path = self.image_paths[idx]

        if self.image_embeddings is not None and image_path in self.image_embeddings:
            image = torch.tensor(self.image_embeddings[image_path], dtype=torch.float32)
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
