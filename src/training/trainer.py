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

    @staticmethod
    def _is_stage1_trainable(name: str) -> bool:
        """Stage 1에서 열 파라미터: LoRA 어댑터 + 각 블록의 **헤드** MLP.

        이전 규칙은 `'lora' in name or 'mlp' in name`이었는데, CLIP 트랜스포머의 각 층에
        `mlp`라는 이름의 feed-forward 서브모듈이 있어서 인코더 내부까지 전부 열렸다.
        생성자에서 `requires_grad = False`로 동결한 것을 여기서 되살려 놓는 셈이라,
        "CLIP은 동결하고 LoRA/헤드만 학습한다"는 설계가 실제로는 지켜지지 않았다:

            query_block  헤드 1,574,656 / LoRA 589,824 / 인코더 내부  56,669,184  ← 열림
            image_block  헤드 1,836,800 / LoRA       0 / 인코더 내부 201,449,472  ← 열림
            text_block   헤드 1,574,656 / LoRA 589,824 / 인코더 내부          0  ← 정상

        SBERT만 무사했던 이유는 내부 층 이름이 intermediate.dense/output.dense라
        'mlp'에 걸리지 않기 때문이다. 결과적으로 CLIP 사전학습 지식이 19만 건·25에폭
        full fine-tuning으로 덮여 catastrophic forgetting 위험이 컸고, Stage 1이
        6에폭부터 과적합(train↓ val↑)한 것도 이와 무관하지 않다.

        헤드 MLP는 `<block>.mlp.`로 시작하고 인코더 내부 FFN은 `...encoder.layers.N.mlp.`
        처럼 중간 경로에 있으므로, 앞 두 조각으로 판별한다.
        """
        if "lora" in name:
            return True
        parts = name.split(".")
        return len(parts) > 1 and parts[1] == "mlp"

    def _log_trainable(self, stage: str) -> None:
        """무엇이 실제로 열렸는지 블록별로 찍는다.

        이름 규칙으로 동결/해제를 정하는 구조라 의도와 실제가 조용히 어긋날 수 있다
        (실제로 'mlp' in name이 CLIP 내부 FFN까지 열어 258M을 full fine-tuning 했다).
        학습 시작 시점에 수치로 남겨두면 다음에는 로그만 봐도 잡힌다.
        """
        by_block: dict[str, int] = {}
        total = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                blk = name.split(".")[0]
                by_block[blk] = by_block.get(blk, 0) + param.numel()
                total += param.numel()
        detail = " / ".join(f"{k} {v:,}" for k, v in sorted(by_block.items()))
        logger.info(f"{stage} - 학습 파라미터 {total:,} ({detail})")

    def _train_stage_1(self, train_loader, val_loader):
        for name, param in self.model.named_parameters():
            param.requires_grad = self._is_stage1_trainable(name)
        self._log_trainable("Stage 1")

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
        """Stage 2 validation: 학습과 같은 InfoNCE(z_content ↔ z_query, in-batch negatives).

        학습 손실과 같은 식이어야 early stopping이 옳은 지점을 고른다. 배치 크기에 따라
        오답 수가 달라져 값이 변하므로, 실행 간 비교는 같은 batch_size에서만 유효하다.
        """
        self.model.eval()
        total_loss = 0.0

        for batch in val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = self.model(batch)
            loss = self.contrastive_loss_fn(outputs['z_content'], outputs['z_query'])
            total_loss += loss.item()

        self.model.train()
        return total_loss / len(val_loader)

    def _train_stage_2(self, train_loader, val_loader):
        for name, param in self.model.named_parameters():
            if "content_block" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self._log_trainable("Stage 2")

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

                # 검색은 z_query·z_content 내적으로 이뤄지므로, 이 단계의 손실은 그 기하를
                # 다듬어야 한다. 배치 안의 다른 아이템을 오답으로 써서 "자기 쿼리에는 가깝게,
                # 나머지 N-1개보다는 멀게"를 학습시킨다.
                #
                # 이전에는 KLDiv(log_softmax(z_content, dim=-1), log_softmax(z_query, dim=-1))
                # 였다 — 후보가 아니라 768개 임베딩 "차원"에 대한 분포를 맞추는 형태다.
                # 차원은 배타적 선택지가 아니라 좌표라서 질문 자체가 성립하지 않고, 수치적으로도
                # L2 정규화된 768차원 벡터는 성분이 ±1/√768≈0.036이라 softmax가 거의 균등해진다.
                # 실측상 완전히 무관한 두 벡터의 KL이 0.00128인데 학습 val loss는 0.00031로,
                # "잘 맞춤"과 "무관함"이 손실 값에서 구분되지 않아 기울기가 소멸했다.
                # 그 결과 content_block이 사실상 학습되지 않았다(아이템 간 평균 코사인
                # 랜덤 초기화 0.352 → 15에폭 후 0.441).
                loss = self.contrastive_loss_fn(z_content, z_query)
                loss.backward()
                optimizer.step()

                self.history.add_stage2_batch(epoch, loss.item())
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

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
                    logger.info(f"Stage 2 - Best checkpoint saved (val_loss={best_val_loss:.4f})")
                else:
                    patience_counter += 1
                    logger.info(f"Stage 2 - No improvement ({patience_counter}/{self.config.early_stopping_patience})")
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Stage 2 - Early stopping at epoch {epoch+1}.")
                        break

        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            logger.info(f"Stage 2 - Loaded best checkpoint (val_loss={best_val_loss:.4f})")
