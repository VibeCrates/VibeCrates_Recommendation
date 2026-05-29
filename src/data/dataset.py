"""
Custom PyTorch Dataset for multi-modal recommendation data.
Handles text, image, and query loading.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict


class MultiModalDataset(Dataset):
    """
    A PyTorch Dataset for multi-modal recommendation data.

    Each sample contains:
    - content_text: Product/content description (string)
    - content_image: Path to image file or PIL Image
    - query: List[str] parsed from DSV (e.g. "q1|q2|q3|q4" → ["q1","q2","q3","q4"])
    """

    def __init__(
        self,
        content_texts: List[str],
        image_paths: List[str],
        queries: List[str],
        image_size: Tuple[int, int] = (224, 224)
    ):
        assert len(content_texts) == len(image_paths) == len(queries), \
            "All input lists must have the same length"

        self.content_texts = content_texts
        self.image_paths = image_paths
        self.queries = queries
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.content_texts)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        content_text = self.content_texts[idx]

        image_path = self.image_paths[idx]
        if str(image_path).startswith("http"):
            import requests
            from io import BytesIO
            r = requests.get(image_path, timeout=10, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"})
            r.raise_for_status()
            image = Image.open(BytesIO(r.content)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)

        # DSV → List[str]; NaN이나 빈 값이면 빈 리스트 대신 원본 문자열 유지
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
    - content_image: List[PIL.Image]
    - query: List[List[str]]  — N queries per item (from DSV)
    """
    return {
        "content_text": [s["content_text"] for s in batch],
        "content_image": [s["content_image"] for s in batch],
        "query": [s["query"] for s in batch],
    }
