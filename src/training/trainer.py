"""
Training pipeline - Main training loop and model training for the two-stage process.
"""
import torch
import torch.optim as optim
import logging
from tqdm import tqdm
from typing import Tuple, Dict

from .config import TrainingConfig
from .losses import InfoNCELoss, KLDivergenceLoss
from src.models.recommender import DualEncoderModel

logger = logging.getLogger(__name__)

class TwoStageTrainer:
    """
    A trainer for the two-stage training process:
    1. Contrastive Learning for Text, Image, and Query encoders.
    2. Knowledge Distillation for the Content encoder.
    """
    def __init__(self, model: DualEncoderModel, config: TrainingConfig, device: torch.device):
        """
        Initialize the trainer.
        
        Args:
            model: The DualEncoderModel to be trained.
            config: The training configuration.
            device: The device to train on (e.g., 'cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.contrastive_loss_fn = InfoNCELoss(temperature=self.config.temperature)
        self.distillation_loss_fn = KLDivergenceLoss()

    def train(self, train_loader, val_loader=None):
        """
        Orchestrates the two-stage training process.
        """
        logger.info("--- Starting Stage 1: Contrastive Learning ---")
        self._train_stage_1(train_loader, val_loader)
        
        logger.info("--- Starting Stage 2: Knowledge Distillation ---")
        self._train_stage_2(train_loader, val_loader)
        
        logger.info("Training finished.")

    def _train_stage_1(self, train_loader, val_loader):
        """
        Executes the contrastive learning stage.
        Trains TextBlock, ImageBlock, and QueryBlock.
        """
        # Set requires_grad for stage 1: train MLPs and LoRA adapters
        for name, param in self.model.named_parameters():
            if 'lora' in name or 'mlp' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        for epoch in range(self.config.num_epochs_stage1):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch+1}/{self.config.num_epochs_stage1}")
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                # Forward pass to get all embeddings
                outputs = self.model(batch)
                z_text, z_image, z_query = outputs['z_text'], outputs['z_image'], outputs['z_query']
                
                # Calculate InfoNCE loss for all pairs
                loss_tq = self.contrastive_loss_fn(z_text, z_query)
                loss_ti = self.contrastive_loss_fn(z_text, z_image)
                loss_iq = self.contrastive_loss_fn(z_image, z_query)
                
                # Total contrastive loss
                loss = loss_tq + loss_ti + loss_iq
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Stage 1 - Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            
            # TODO: Add validation loop if val_loader is provided

    def _train_stage_2(self, train_loader, val_loader):
        """
        Executes the knowledge distillation stage.
        Trains only the ContentBlock.
        """
        # Set requires_grad for stage 2: only train ContentBlock
        for name, param in self.model.named_parameters():
            if 'content_block' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        for epoch in range(self.config.num_epochs_stage2):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch+1}/{self.config.num_epochs_stage2}")
            for batch in progress_bar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                # Forward pass to get content and query vectors
                outputs = self.model(batch)
                z_content, z_query = outputs['z_content'], outputs['z_query']
                
                # The "teacher" (z_query) should not have gradients flowing back to it
                with torch.no_grad():
                    teacher_dist = F.log_softmax(z_query, dim=-1)
                
                # The "student" (z_content)
                student_dist = F.log_softmax(z_content, dim=-1)
                
                # Calculate KL-Divergence loss
                loss = self.distillation_loss_fn(student_dist, teacher_dist)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
                
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Stage 2 - Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            
            # TODO: Add validation loop if val_loader is provided
