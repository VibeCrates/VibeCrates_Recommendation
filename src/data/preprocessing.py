"""
Preprocessing - Data preprocessing and feature engineering
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List


class DataPreprocessor:
    """
    TODO: Implement data preprocessing pipeline.
    
    Should handle:
    - Missing value imputation
    - Categorical encoding
    - Feature scaling
    - Outlier detection/removal
    - Train-test splitting
    """
    
    def __init__(self):
        """Initialize preprocessor."""
        self.preprocessors = {}
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """
        TODO: Handle missing values in data.
        
        Args:
            df: DataFrame with potential missing values
            strategy: Strategy for handling missing values ('mean', 'median', 'drop', 'forward_fill', etc.)
            
        Returns:
            DataFrame with missing values handled
        """
        pass
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                   categorical_cols: List[str]) -> pd.DataFrame:
        """
        TODO: Encode categorical features.
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names
            
        Returns:
            DataFrame with encoded categorical features
        """
        # TODO: Use OneHotEncoder, LabelEncoder, or TargetEncoder
        pass
    
    def normalize_numerical_features(self, X: np.ndarray, 
                                    method: str = "standardization") -> Tuple[np.ndarray, object]:
        """
        TODO: Normalize numerical features.
        
        Args:
            X: Feature matrix
            method: Normalization method ('standardization', 'min-max', etc.)
            
        Returns:
            Tuple of (normalized_data, scaler_object)
        """
        pass
    
    def create_user_profiles(self, interaction_df: pd.DataFrame) -> pd.DataFrame:
        """
        TODO: Create aggregated user profiles from interactions.
        
        Args:
            interaction_df: DataFrame with user-item interactions
            
        Returns:
            DataFrame with user profiles
        """
        pass
    
    def create_item_features(self, item_df: pd.DataFrame) -> np.ndarray:
        """
        TODO: Create feature representations for items.
        
        Args:
            item_df: DataFrame with item information
            
        Returns:
            Feature matrix for items
        """
        pass


def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for duplication
        
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset)


def remove_outliers(X: np.ndarray, method: str = "iqr", 
                   threshold: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO: Detect and remove outliers.
    
    Args:
        X: Feature matrix
        method: Method for outlier detection ('iqr', 'z-score', 'isolation-forest')
        threshold: Threshold for outlier detection
        
    Returns:
        Tuple of (cleaned_data, outlier_mask)
    """
    pass


def resample_data(X: np.ndarray, y: np.ndarray = None, 
                 strategy: str = "oversample") -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO: Handle class imbalance through resampling.
    
    Args:
        X: Feature matrix
        y: Target labels
        strategy: Resampling strategy ('oversample', 'undersample', 'smote')
        
    Returns:
        Resampled data and labels
    """
    pass
