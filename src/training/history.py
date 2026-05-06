"""
Training history tracking for monitoring and visualization.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List


class TrainingHistory:
    """
    Tracks training metrics across epochs and stages.
    """
    
    def __init__(self):
        """Initialize history tracking."""
        self.history = {
            'stage1': {
                'epoch': [],
                'loss_tq': [],
                'loss_ti': [],
                'loss_iq': [],
                'total_loss': [],
                'val_loss': []
            },
            'stage2': {
                'epoch': [],
                'loss': [],
                'val_loss': []
            }
        }
    
    def add_stage1_batch(self, epoch: int, loss_tq: float, loss_ti: float, loss_iq: float, total_loss: float):
        """
        Record Stage 1 batch metrics.
        
        Args:
            epoch: Current epoch
            loss_tq: Text-Query loss
            loss_ti: Text-Image loss
            loss_iq: Image-Query loss
            total_loss: Total loss
        """
        self.history['stage1']['epoch'].append(epoch)
        self.history['stage1']['loss_tq'].append(loss_tq)
        self.history['stage1']['loss_ti'].append(loss_ti)
        self.history['stage1']['loss_iq'].append(loss_iq)
        self.history['stage1']['total_loss'].append(total_loss)
    
    def add_stage1_val(self, epoch: int, val_loss: float):
        """
        Record Stage 1 validation loss.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
        """
        self.history['stage1']['val_loss'].append(val_loss)
    
    def add_stage2_batch(self, epoch: int, loss: float):
        """
        Record Stage 2 batch metrics.
        
        Args:
            epoch: Current epoch
            loss: KL-Divergence loss
        """
        self.history['stage2']['epoch'].append(epoch)
        self.history['stage2']['loss'].append(loss)
    
    def add_stage2_val(self, epoch: int, val_loss: float):
        """
        Record Stage 2 validation loss.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
        """
        self.history['stage2']['val_loss'].append(val_loss)
    
    def get_stage1_loss_per_epoch(self) -> tuple:
        """
        Calculate average loss per epoch for Stage 1.
        
        Returns:
            Tuple of (epochs, avg_losses)
        """
        if not self.history['stage1']['epoch']:
            return [], []
        
        epochs_array = np.array(self.history['stage1']['epoch'])
        losses_array = np.array(self.history['stage1']['total_loss'])
        
        unique_epochs = np.unique(epochs_array)
        avg_losses = []
        
        for epoch in unique_epochs:
            mask = epochs_array == epoch
            avg_loss = losses_array[mask].mean()
            avg_losses.append(avg_loss)
        
        return unique_epochs.tolist(), avg_losses
    
    def get_stage2_loss_per_epoch(self) -> tuple:
        """
        Calculate average loss per epoch for Stage 2.
        
        Returns:
            Tuple of (epochs, avg_losses)
        """
        if not self.history['stage2']['epoch']:
            return [], []
        
        epochs_array = np.array(self.history['stage2']['epoch'])
        losses_array = np.array(self.history['stage2']['loss'])
        
        unique_epochs = np.unique(epochs_array)
        avg_losses = []
        
        for epoch in unique_epochs:
            mask = epochs_array == epoch
            avg_loss = losses_array[mask].mean()
            avg_losses.append(avg_loss)
        
        return unique_epochs.tolist(), avg_losses
    
    def save(self, filepath: str):
        """
        Save training history to JSON file.
        
        Args:
            filepath: Path to save the history
        """
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for stage, metrics in self.history.items():
            history_dict[stage] = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    history_dict[stage][key] = value.tolist()
                else:
                    history_dict[stage][key] = value
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)
    
    @staticmethod
    def load(filepath: str) -> 'TrainingHistory':
        """
        Load training history from JSON file.
        
        Args:
            filepath: Path to load the history from
            
        Returns:
            TrainingHistory object
        """
        history_obj = TrainingHistory()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        history_obj.history = data
        return history_obj
