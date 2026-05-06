"""
Base model class - Abstract base class for all recommendation models
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import numpy as np


class BaseRecommender(ABC):
    """
    Abstract base class for recommendation models.
    
    All recommendation models should inherit from this class and implement
    the abstract methods: fit, predict, and evaluate.
    """
    
    def __init__(self, name: str = "BaseRecommender"):
        """
        Initialize the base recommender.
        
        Args:
            name: Name of the model
        """
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        """
        Train the recommendation model.
        
        Args:
            X: Training features/data
            y: Target labels (if supervised learning)
            **kwargs: Additional arguments for training
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, top_k: int = 10) -> np.ndarray:
        """
        Generate recommendations for given input.
        
        Args:
            X: Input data for prediction
            top_k: Number of top recommendations to return
            
        Returns:
            Array of recommendations
        """
        pass
    
    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray = None, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features/data
            y_test: Test labels (if applicable)
            **kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path where the model should be saved
        """
        raise NotImplementedError("Subclass must implement save method")
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        raise NotImplementedError("Subclass must implement load method")
