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
    - query: Search query (string)
    """
    
    def __init__(
        self,
        content_texts: List[str],
        image_paths: List[str],
        queries: List[str],
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the dataset.
        
        Args:
            content_texts: List of content text descriptions
            image_paths: List of paths to images
            queries: List of search queries
            image_size: Target image size for resizing (height, width)
        """
        assert len(content_texts) == len(image_paths) == len(queries), \
            "All input lists must have the same length"
        
        self.content_texts = content_texts
        self.image_paths = image_paths
        self.queries = queries
        self.image_size = image_size
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.content_texts)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
            - content_text: Text description
            - content_image: PIL Image (224x224)
            - query: Search query text
        """
        # Load text (keep as string for TextBlock to process)
        content_text = self.content_texts[idx]
        
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Load query (keep as string for QueryBlock to process)
        query = self.queries[idx]
        
        return {
            "content_text": content_text,
            "content_image": image,
            "query": query
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching.
    
    Converts a list of samples into a batch:
    - content_text: List of strings (for SBERT)
    - content_image: List of PIL Images (for CLIP)
    - query: List of strings (for CLIP Text Encoder)
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Dictionary with batched data
    """
    content_texts = [sample["content_text"] for sample in batch]
    content_images = [sample["content_image"] for sample in batch]
    queries = [sample["query"] for sample in batch]
    
    return {
        "content_text": content_texts,
        "content_image": content_images,
        "query": queries
    }
