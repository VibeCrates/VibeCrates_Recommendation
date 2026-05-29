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
        (X_train, X_test, y_train, y_test) or (X_train, X_test) if y is None
    """
    from sklearn.model_selection import train_test_split
    if y is None:
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        return X_train, X_test
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, object]:
    """
    Normalize features using StandardScaler.

    Args:
        X: Feature data to normalize

    Returns:
        (normalized_X, fitted_scaler)
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def calculate_similarity(X: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Calculate pairwise similarity matrix.

    Args:
        X: (N, D) data matrix
        metric: "cosine" | "euclidean" | "pearson"

    Returns:
        (N, N) similarity matrix. cosine: [-1, 1], euclidean/pearson: [0, 1].
        값이 클수록 유사.
    """
    if metric == "cosine":
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(X)

    if metric == "euclidean":
        from sklearn.metrics.pairwise import euclidean_distances
        dist = euclidean_distances(X)
        return 1.0 / (1.0 + dist)

    if metric == "pearson":
        # 행 단위 z-score 후 코사인 유사도 = 피어슨 상관계수
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-8
        X_norm = (X - mean) / std
        sim = X_norm @ X_norm.T / X.shape[1]
        return (sim + 1.0) / 2.0  # [-1, 1] → [0, 1]

    raise ValueError(f"Unknown metric: {metric!r}. 선택 가능: cosine | euclidean | pearson")
