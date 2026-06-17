"""
Training pipeline - Main training loop and model training for the two-stage process.
"""
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
from tqdm import tqdm

from .config import TrainingConfig
from .losses import InfoNCELoss, KLDivergenceLoss
from .history import TrainingHistory
from src.models.recommender import DualEncoderModel

logger = logging.getLogger(__name__)

class TwoStageTrainer:
    def __init__(self, model: DualEncoderModel, config: TrainingConfig, device: torch.device,
                 checkpoint_dir: str = "models"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.contrastive_loss_fn = InfoNCELoss(temperature=self.config.temperature)
        self.distillation_loss_fn = KLDivergenceLoss()
        self.history = TrainingHistory()
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train(self, train_loader, val_loader=None):
        """
        Orchestrates the two-stage training process.
        
        Returns:
            TrainingHistory object containing all recorded metrics
        """
        logger.info("--- Starting Stage 1: Contrastive Learning ---")
        self._train_stage_1(train_loader, val_loader)
        
        logger.info("--- Starting Stage 2: Knowledge Distillation ---")
        self._train_stage_2(train_loader, val_loader)
        
        logger.info("Training finished.")
        return self.history

    def _train_stage_1(self, train_loader, val_loader):
        for name, param in self.model.named_parameters():
            if 'lora' in name or 'mlp' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        ckpt_path = os.path.join(self.checkpoint_dir, "best_stage1.pt")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.num_epochs_stage1):
            self.model.train()
            total_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch+1}/{self.config.num_epochs_stage1}")
            for batch in progress_bar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                optimizer.zero_grad()

                outputs = self.model(batch)
                z_text, z_image, z_query = outputs["z_text"], outputs["z_image"], outputs["z_query"]

                loss_tq = self.contrastive_loss_fn(z_text, z_query)
                loss_ti = self.contrastive_loss_fn(z_text, z_image)
                loss_iq = self.contrastive_loss_fn(z_image, z_query)
                loss = loss_tq + loss_ti + loss_iq

                loss.backward()
                optimizer.step()

                self.history.add_stage1_batch(epoch, loss_tq.item(), loss_ti.item(), loss_iq.item(), loss.item())
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Stage 1 - Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

            if val_loader is not None:
                val_loss = self._validate_stage_1(val_loader)
                self.history.add_stage1_val(epoch, val_loss)
                logger.info(f"Stage 1 - Epoch {epoch+1} Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), ckpt_path)
                    logger.info(f"Stage 1 - Best checkpoint saved (val_loss={best_val_loss:.4f})")
                else:
                    patience_counter += 1
                    logger.info(f"Stage 1 - No improvement ({patience_counter}/{self.config.early_stopping_patience})")
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Stage 1 - Early stopping at epoch {epoch+1}.")
                        break

        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            logger.info(f"Stage 1 - Loaded best checkpoint (val_loss={best_val_loss:.4f})")

    @torch.no_grad()
    def _validate_stage_1(self, val_loader) -> float:
        """Stage 1 validation: sum of InfoNCE losses across text-query, text-image, image-query pairs."""
        self.model.eval()
        total_loss = 0.0

        for batch in val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = self.model(batch)
            z_text, z_image, z_query = outputs['z_text'], outputs['z_image'], outputs['z_query']

            loss = (self.contrastive_loss_fn(z_text, z_query)
                    + self.contrastive_loss_fn(z_text, z_image)
                    + self.contrastive_loss_fn(z_image, z_query))
            total_loss += loss.item()

        self.model.train()
        return total_loss / len(val_loader)

    @torch.no_grad()
    def _validate_stage_2(self, val_loader) -> float:
        """Stage 2 validation: KL-Divergence between z_content (student) and z_query (teacher) distributions."""
        self.model.eval()
        total_loss = 0.0

        for batch in val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = self.model(batch)
            z_content, z_query = outputs['z_content'], outputs['z_query']

            teacher_dist = F.log_softmax(F.normalize(z_query, p=2, dim=-1), dim=-1)
            student_dist = F.log_softmax(z_content, dim=-1)
            loss = self.distillation_loss_fn(student_dist, teacher_dist)
            total_loss += loss.item()

        self.model.train()
        return total_loss / len(val_loader)

    def _train_stage_2(self, train_loader, val_loader):
        for name, param in self.model.named_parameters():
            if "content_block" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate_stage2,
            weight_decay=self.config.weight_decay,
        )

        ckpt_path = os.path.join(self.checkpoint_dir, "best_stage2.pt")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.num_epochs_stage2):
            self.model.train()
            total_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch+1}/{self.config.num_epochs_stage2}")
            for batch in progress_bar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                optimizer.zero_grad()

                outputs = self.model(batch)
                z_content, z_query = outputs["z_content"], outputs["z_query"]

                with torch.no_grad():
                    teacher_dist = F.log_softmax(F.normalize(z_query, p=2, dim=-1), dim=-1)
                student_dist = F.log_softmax(z_content, dim=-1)

                loss = self.distillation_loss_fn(student_dist, teacher_dist)
                loss.backward()
                optimizer.step()

                self.history.add_stage2_batch(epoch, loss.item())
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Stage 2 - Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

            if val_loader is not None:
                val_loss = self._validate_stage_2(val_loader)
                self.history.add_stage2_val(epoch, val_loss)
                logger.info(f"Stage 2 - Epoch {epoch+1} Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), ckpt_path)
                    logger.info(f"Stage 2 - Best checkpoint saved (val_loss={best_val_loss:.6f})")
                else:
                    patience_counter += 1
                    logger.info(f"Stage 2 - No improvement ({patience_counter}/{self.config.early_stopping_patience})")
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Stage 2 - Early stopping at epoch {epoch+1}.")
                        break

        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            logger.info(f"Stage 2 - Loaded best checkpoint (val_loss={best_val_loss:.6f})")
