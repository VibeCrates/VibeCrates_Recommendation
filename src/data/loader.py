"""
Data loader - Load data from various sources
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """
    TODO: Implement data loader for loading recommendation system data.
    
    This should handle:
    - Loading from CSV files
    - Loading from databases
    - Loading from APIs
    - Data validation
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path) if data_path else Path("data/processed")
    
    def load_user_data(self, filename: str = "users.csv") -> pd.DataFrame:
        """
        TODO: Load user data from file.
        
        Args:
            filename: Name of the user data file
            
        Returns:
            DataFrame with user information
        """
        pass
    
    def load_item_data(self, filename: str = "items.csv") -> pd.DataFrame:
        """
        TODO: Load item data from file.
        
        Args:
            filename: Name of the item data file
            
        Returns:
            DataFrame with item information
        """
        pass
    
    def load_interaction_data(self, filename: str = "interactions.csv") -> pd.DataFrame:
        """
        TODO: Load user-item interaction data.
        
        Args:
            filename: Name of the interaction data file
            
        Returns:
            DataFrame with user-item interactions (ratings, clicks, purchases, etc.)
        """
        pass
    
    def load_features(self, filename: str = "features.npy") -> np.ndarray:
        """
        TODO: Load preprocessed feature matrices.
        
        Args:
            filename: Name of the feature file
            
        Returns:
            Numpy array of features
        """
        pass


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load CSV file into DataFrame.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Pandas DataFrame
    """
    return pd.read_csv(filepath)


def load_numpy(filepath: str) -> np.ndarray:
    """
    Load numpy array from file.
    
    Args:
        filepath: Path to numpy file (.npy or .npz)
        
    Returns:
        Numpy array
    """
    return np.load(filepath)


def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """
    Save DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        filepath: Path where to save the CSV
    """
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def save_numpy(arr: np.ndarray, filepath: str) -> None:
    """
    Save numpy array to file.
    
    Args:
        arr: Array to save
        filepath: Path where to save the numpy file
    """
    np.save(filepath, arr)
    print(f"Array saved to {filepath}")
