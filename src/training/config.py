"""
Training configuration - Hyperparameters and training settings
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Configuration for model training.
    
    TODO: Adjust these parameters based on your model and dataset
    """
    
    # Model parameters
    model_type: str = "collaborative_filtering"  # Type of model to train
    
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Data parameters
    train_test_split: float = 0.8
    validation_split: float = 0.1
    random_seed: int = 42
    
    # Regularization
    dropout_rate: float = 0.2
    l1_reg: float = 0.0
    l2_reg: float = 0.0
    
    # Model architecture (for neural networks)
    hidden_dims: tuple = (128, 64, 32)
    embedding_dim: int = 64
    
    # Training behavior
    early_stopping_patience: int = 5
    verbose: bool = True
    log_interval: int = 100
    
    # Device
    device: str = "cpu"  # "cpu" or "cuda"
    
    def to_dict(self):
        """Convert config to dictionary."""
        return self.__dict__


def load_training_config(config_path: str = None) -> TrainingConfig:
    """
    TODO: Load training configuration from file (JSON, YAML, etc.).
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        TrainingConfig object
    """
    pass


def save_training_config(config: TrainingConfig, output_path: str) -> None:
    """
    TODO: Save training configuration to file.
    
    Args:
        config: TrainingConfig object to save
        output_path: Path where to save the configuration
    """
    pass
