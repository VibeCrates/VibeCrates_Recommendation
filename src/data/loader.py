"""
Data loader utilities for loading multi-modal recommendation data.
"""
import random
from typing import Dict, List, Tuple, Union

import torch
from torch.utils.data import DataLoader, Sampler, random_split
from pathlib import Path
import pandas as pd
import logging

from .dataset import MultiModalDataset, collate_fn

logger = logging.getLogger(__name__)


class DomainBalancedBatchSampler(Sampler[List[int]]):
    """매 배치가 세 도메인을 모두 포함하도록 구성한다 (4-2, 2026-09-17 — worklist 4-2).

    왜 필요한가: centering(인덱스에서 도메인 중심 벡터를 사후에 빼는 방법)이 실패한 이유는
    도메인마다 다른 강도로 작용했기 때문이다(공통 성분이 가장 큰 music이 가장 많이
    변해 유용한 신호까지 잃었다, 판정 0.754 → 0.723). 사후 조작 대신 학습 쪽에서 원인을
    끊는다 — book이 훈련셋의 58%라 무작위 셔플로는 배치 안 negative가 대부분 book끼리고,
    InfoNCE가 도메인 간 경계를 거의 학습하지 못한다.

    도메인별 인덱스를 따로 섞어 두고 배치마다 (거의) 동수로 뽑는다. movie/music처럼 작은
    도메인은 한 에폭 안에서 여러 번 재순회되고(oversample), book은 한 번 다 돌면 그걸로
    에폭이 끝난다. 배치 수는 기존(무작위 셔플)과 같게 `총 행 수 // batch_size`로 맞춰
    한 에폭의 스텝 수·연산량이 baseline과 같도록 했다 — 배치 구성만 바뀌고 학습 비용은
    그대로다.
    """

    def __init__(self, domain_labels: List[str], batch_size: int, seed: int = 0):
        self.batch_size = batch_size
        self.seed = seed
        self.by_domain: Dict[str, List[int]] = {}
        for i, d in enumerate(domain_labels):
            self.by_domain.setdefault(d, []).append(i)
        self.domains = sorted(self.by_domain)
        if len(self.domains) < 2:
            raise ValueError("도메인이 1개뿐이면 균형 배치가 의미 없다 — 대상 df를 확인할 것.")

        base = batch_size // len(self.domains)
        rem = batch_size - base * len(self.domains)
        # 나머지는 앞쪽 도메인(사전순)부터 하나씩 — 매 배치 동일하므로 결정적이다.
        self.per_domain = [base + (1 if i < rem else 0) for i in range(len(self.domains))]
        self.n_batches = sum(len(v) for v in self.by_domain.values()) // batch_size
        self._epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        pools = {d: idxs[:] for d, idxs in self.by_domain.items()}
        for p in pools.values():
            rng.shuffle(p)
        cursors = {d: 0 for d in self.domains}

        def draw(d: str, k: int) -> List[int]:
            pool, c, out = pools[d], cursors[d], []
            for _ in range(k):
                if c >= len(pool):
                    rng.shuffle(pool)
                    c = 0
                out.append(pool[c])
                c += 1
            cursors[d] = c
            return out

        for _ in range(self.n_batches):
            batch = []
            for d, k in zip(self.domains, self.per_domain):
                batch.extend(draw(d, k))
            rng.shuffle(batch)          # 도메인끼리 뭉쳐 정렬되지 않도록
            yield batch

    def __len__(self) -> int:
        return self.n_batches


def load_data_from_csv(
    csv_path: Union[str, Path],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42
) -> Tuple[MultiModalDataset, MultiModalDataset, MultiModalDataset]:
    """
    Load data from a CSV file and split into train/val/test sets.
    
    CSV format expected:
    content_text,image_path,query
    "Product description","path/to/image.jpg","search query"
    
    Args:
        csv_path: Path to CSV file
        test_size: Fraction of data to use for testing (0.2 = 20%)
        val_size: Fraction of data to use for validation (0.1 = 10%)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Verify required columns
    required_cols = ['content_text', 'image_path', 'query']
    assert all(col in df.columns for col in required_cols), \
        f"CSV must contain columns: {required_cols}"
    
    # Extract lists
    content_texts = df['content_text'].tolist()
    image_paths = df['image_path'].tolist()
    queries = df['query'].tolist()
    
    # Create full dataset
    full_dataset = MultiModalDataset(content_texts, image_paths, queries)
    
    # Calculate split sizes
    test_count = int(len(full_dataset) * test_size)
    val_count = int((len(full_dataset) - test_count) * val_size)
    train_count = len(full_dataset) - test_count - val_count
    
    # Split dataset
    torch.manual_seed(random_seed)
    train_dataset, remaining = random_split(
        full_dataset,
        [train_count, test_count + val_count]
    )
    val_dataset, test_dataset = random_split(
        remaining,
        [val_count, test_count]
    )
    
    logger.info(f"Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: MultiModalDataset,
    val_dataset: MultiModalDataset = None,
    test_dataset: MultiModalDataset = None,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> dict:
    """
    Create PyTorch DataLoaders for train/val/test datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Dictionary containing dataloaders
    """
    loaders = {}
    
    # Train loader
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Validation loader
    if val_dataset is not None:
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    # Test loader
    if test_dataset is not None:
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    return loaders


def get_dataloaders_from_df(
    df: pd.DataFrame,
    batch_size: int = 32,
    test_size: float = 0.2,
    val_size: float = 0.1,
    num_workers: int = 0,
    random_seed: int = 42,
    image_embeddings=None,
    sample_one_query: bool = False,
    title_dropout: float = 0.0,
    label_dropout: float = 0.0,
    domain_balanced_batches: bool = False,
) -> dict:
    """
    Create DataLoaders directly from a DataFrame (already loaded / filtered).

    Expects columns: content_text, image_path, query (+ "domain" if
    domain_balanced_batches=True).
    """
    required = ["content_text", "image_path", "query"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    if domain_balanced_batches and "domain" not in df.columns:
        raise ValueError("domain_balanced_batches=True인데 df에 'domain' 컬럼이 없다.")

    full_dataset = MultiModalDataset(
        df["content_text"].tolist(),
        df["image_path"].tolist(),
        df["query"].tolist(),
        image_embeddings=image_embeddings,
        sample_one_query=sample_one_query,
        title_dropout=title_dropout,
        label_dropout=label_dropout,
    )

    n = len(full_dataset)
    test_count = int(n * test_size)
    val_count = int((n - test_count) * val_size)
    train_count = n - test_count - val_count

    torch.manual_seed(random_seed)
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_count, val_count, test_count])

    logger.info(f"Split: train={train_count:,} val={val_count:,} test={test_count:,}")

    if not domain_balanced_batches:
        return create_dataloaders(train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=num_workers)

    # train만 도메인 균형 배치로, val/test는 순서가 중요하지 않으니 기존 방식 그대로.
    all_domains = df["domain"].tolist()
    train_domains = [all_domains[i] for i in train_ds.indices]
    sampler = DomainBalancedBatchSampler(train_domains, batch_size, seed=random_seed)
    logger.info(f"Domain-balanced batches: {dict(zip(sampler.domains, sampler.per_domain))} "
                f"per batch, {sampler.n_batches:,} batches/epoch")

    loaders = {
        "train": DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=True),
    }
    if val_ds is not None:
        loaders["val"] = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    if test_ds is not None:
        loaders["test"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    return loaders


def get_dataloaders_from_csv(
    csv_path: Union[str, Path],
    batch_size: int = 32,
    test_size: float = 0.2,
    val_size: float = 0.1,
    num_workers: int = 0,
    random_seed: int = 42
) -> dict:
    """
    Convenience function to load data from CSV and create dataloaders in one step.
    
    Args:
        csv_path: Path to CSV file
        batch_size: Batch size
        test_size: Fraction for test set
        val_size: Fraction for validation set
        num_workers: Number of workers
        random_seed: Random seed
        
    Returns:
        Dictionary containing dataloaders
    """
    train_dataset, val_dataset, test_dataset = load_data_from_csv(
        csv_path,
        test_size=test_size,
        val_size=val_size,
        random_seed=random_seed
    )
    
    loaders = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return loaders
