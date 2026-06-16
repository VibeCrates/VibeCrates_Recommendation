"""
Training configuration - Hyperparameters and training settings
"""
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class TrainingConfig:
    """
    Configuration for the two-stage model training.
    """
    # --- General Parameters ---
    device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    random_seed: int = 42

    # --- Stage 1: Contrastive Learning Parameters ---
    num_epochs_stage1: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    temperature: float = 0.07  # For InfoNCE Loss
    
    # --- Stage 2: Distillation Learning Parameters ---
    num_epochs_stage2: int = 15
    # You might want a different learning rate for distillation
    learning_rate_stage2: float = 5e-5 

    # --- Data Parameters ---
    batch_size: int = 32
    train_test_split: float = 0.8
    validation_split: float = 0.1
    
    # --- Model Architecture ---
    # These are mostly fixed by the pre-trained models but can be here for reference
    embedding_dim: int = 768
    
    # --- LoRA Configuration (for TextBlock) ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q", "v"])

    # --- Training Behavior ---
    early_stopping_patience: int = 3
    verbose: bool = True
    log_interval: int = 100

    def to_dict(self):
        """Convert config to dictionary."""
        return self.__dict__

# Functions to load/save config from/to a file (e.g., YAML or JSON) can be added here.
# For simplicity, we are using a dataclass directly.
