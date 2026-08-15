"""
Main training script for the recommendation model.
Orchestrates data loading, model initialization, and training pipeline.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import logging
from pathlib import Path
from argparse import ArgumentParser

from src.training.config import TrainingConfig
from src.training.trainer import TwoStageTrainer
from src.models.recommender import DualEncoderModel
import pandas as pd
from src.data.loader import get_dataloaders_from_df
from scripts.build_index import build_and_save

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """
    Main training function.
    """
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Setup config
    config = TrainingConfig(
        device=str(device),
        num_epochs_stage1=args.epochs_stage1,
        num_epochs_stage2=args.epochs_stage2,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
    )
    logger.info(f"Training config: {config.to_dict()}")
    
    # Load data
    logger.info(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    if args.domain:
        df = df[df["domain"] == args.domain].reset_index(drop=True)
        logger.info(f"Filtered to domain='{args.domain}': {len(df):,} rows")

    image_embeddings = None
    if args.image_embeddings:
        logger.info(f"Loading image embeddings from {args.image_embeddings}...")
        image_embeddings = torch.load(args.image_embeddings, weights_only=False)
        logger.info(f"Loaded {len(image_embeddings):,} embeddings")

    dataloaders = get_dataloaders_from_df(
        df=df,
        batch_size=config.batch_size,
        test_size=1.0 - config.train_test_split,
        val_size=config.validation_split,
        num_workers=args.num_workers,
        random_seed=config.random_seed,
        image_embeddings=image_embeddings,
        sample_one_query=args.sample_one_query,
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders.get('val')
    logger.info(f"Data loaded. Train batches: {len(train_loader)}")
    
    # Initialize model
    logger.info("Initializing DualEncoderModel...")
    model = DualEncoderModel(query_lora=args.query_lora)
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    checkpoint_dir = str(Path(args.save_path).parent)
    trainer = TwoStageTrainer(model, config, device, checkpoint_dir=checkpoint_dir)
    
    # Train
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Save history for visualization
    if args.history_path:
        history_path = Path(args.history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history.save(str(history_path))
        logger.info(f"Training history saved to {history_path}")
    
    # Save model
    if args.save_path:
        model_save_path = Path(args.save_path)
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model saved to {model_save_path}")
    
    logger.info("Training completed!")

    if args.skip_index:
        logger.info("Index build skipped (--skip-index).")
        return

    # Build item index from best model weights.
    # image_embeddings를 반드시 넘긴다 — 넘기지 않으면 MultiModalDataset이 PIL 경로로
    # 떨어져 학습과 다른 이미지를 쓰고(로컬 파일 없는 항목은 학습 시 0 벡터인데 여기서는
    # http로 내려받는다), 다운로드 때문에 몇 시간이 걸린다. 8/14 실행에서 music 인덱스
    # 하나가 2시간 넘게 돌다 중단됐다 — 임베딩을 넘기면 4분이다.
    logger.info("Building item index...")
    index_dir = args.index_dir
    domains = [args.domain] if args.domain else ["movie", "music", "book"]
    for domain in domains:
        try:
            build_and_save(domain, model, device, batch_size=args.batch_size,
                           index_dir=index_dir, image_embeddings=image_embeddings)
        except Exception as e:
            logger.warning(f"[{domain}] Index build failed: {e}")
    logger.info("Index build complete.")


if __name__ == '__main__':
    parser = ArgumentParser(description='Train the recommendation model')
    
    # Data arguments
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/sample_data.csv',
        help='Path to CSV file with training data'
    )
    parser.add_argument(
        '--image-embeddings',
        type=str,
        default=None,
        help='Path to pre-computed image embeddings (.pt file from extract_image_embeddings.py)'
    )
    parser.add_argument(
        '--domain',
        type=str,
        choices=['movie', 'music', 'book'],
        default=None,
        help='Filter training data to a single domain (default: use all domains in the CSV)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Number of data loading workers'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--epochs-stage1',
        type=int,
        default=10,
        help='Number of epochs for stage 1 (contrastive learning)'
    )
    parser.add_argument(
        '--epochs-stage2',
        type=int,
        default=15,
        help='Number of epochs for stage 2 (distillation)'
    )
    parser.add_argument(
        '--skip-index', action='store_true',
        help='학습만 하고 인덱스는 만들지 않는다. 스크립트가 build_index.py로 따로 만들 때 쓴다.'
    )
    parser.add_argument(
        '--query-lora', action='store_true',
        help='개선안 1: QueryBlock의 CLIP 텍스트 인코더에 LoRA를 붙여 쿼리 쪽도 적응 가능하게 '
             '한다(콘텐츠 TextBlock과 대칭). state_dict 키가 달라지므로 이 플래그로 만든 '
             '체크포인트는 build_index/eval에서도 같은 플래그로 읽어야 한다.'
    )
    parser.add_argument(
        '--sample-one-query', action='store_true',
        help='개선안 2: 학습 시 아이템당 DSV 쿼리 3개 중 1개를 매 스텝 무작위로 뽑는다. '
             '끄면 3개를 mean-pool 해 콘텐츠가 세 방향의 타협 centroid에 정렬된다.'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=3,
        help='개선 없는 에폭이 이만큼 연속되면 중단. 각 스테이지의 best 체크포인트는 '
             'models/best_stage{1,2}.pt에 저장되고 스테이지 종료 시 되불러온다.'
    )
    
    # Device arguments
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device to use for training'
    )
    
    # Model saving
    parser.add_argument(
        '--save-path',
        type=str,
        default='models/trained_model.pt',
        help='Path to save the trained model'
    )
    parser.add_argument(
        '--history-path',
        type=str,
        default='logs/training_history.json',
        help='Path to save training history (loss curves)'
    )
    parser.add_argument(
        '--index-dir',
        type=str,
        default='indexes',
        help='Directory to save pre-computed item embeddings'
    )
    
    args = parser.parse_args()
    
    main(args)
