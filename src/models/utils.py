"""
Model utility functions
"""
import numpy as np
from typing import Tuple, List
import joblib
from pathlib import Path


def save_model(model: object, filepath: str) -> None:
    """
    Save model to disk using joblib.
    
    Args:
        model: Model object to save
        filepath: Path where model should be saved
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> object:
    """
    Load model from disk using joblib.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def split_train_test(X: np.ndarray, y: np.ndarray = None, test_size: float = 0.2, 
                     random_state: int = None) -> Tuple:
    """
    Split data into train and test sets.
    
    Args:
        X: Feature data
        y: Target data (optional)
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) or (X_train, X_test) if y is None
    """
    # TODO: Use sklearn's train_test_split or implement custom split
    pass


def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, object]:
    """
    Normalize features using StandardScaler or similar.
    
    Args:
        X: Feature data to normalize
        
    Returns:
        Tuple of (normalized_data, scaler_object)
    """
    # TODO: Implement normalization using sklearn.preprocessing
    pass


def calculate_similarity(X: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Calculate similarity matrix between items or users.
    
    Args:
        X: Data matrix
        metric: Similarity metric ('cosine', 'euclidean', 'pearson', etc.)
        
    Returns:
        Similarity matrix
    """
    # TODO: Implement similarity calculation using sklearn.metrics.pairwise
    pass
