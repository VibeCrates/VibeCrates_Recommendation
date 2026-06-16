"""
Data loader utilities for loading multi-modal recommendation data.
"""
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import Tuple, Union
import pandas as pd
import logging

from .dataset import MultiModalDataset, collate_fn

logger = logging.getLogger(__name__)


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
) -> dict:
    """
    Create DataLoaders directly from a DataFrame (already loaded / filtered).

    Expects columns: content_text, image_path, query
    """
    required = ["content_text", "image_path", "query"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    full_dataset = MultiModalDataset(
        df["content_text"].tolist(),
        df["image_path"].tolist(),
        df["query"].tolist(),
    )

    n = len(full_dataset)
    test_count = int(n * test_size)
    val_count = int((n - test_count) * val_size)
    train_count = n - test_count - val_count

    torch.manual_seed(random_seed)
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_count, val_count, test_count])

    logger.info(f"Split: train={train_count:,} val={val_count:,} test={test_count:,}")
    return create_dataloaders(train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=num_workers)


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
