"""
Main training script for the recommendation model.
Orchestrates data loading, model initialization, and training pipeline.
"""
import torch
import logging
from pathlib import Path
from argparse import ArgumentParser

from src.training.config import TrainingConfig
from src.training.trainer import TwoStageTrainer
from src.models.recommender import DualEncoderModel
import pandas as pd
from src.data.loader import get_dataloaders_from_df

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
        learning_rate=args.learning_rate
    )
    logger.info(f"Training config: {config.to_dict()}")
    
    # Load data
    logger.info(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    if args.domain:
        df = df[df["domain"] == args.domain].reset_index(drop=True)
        logger.info(f"Filtered to domain='{args.domain}': {len(df):,} rows")
    dataloaders = get_dataloaders_from_df(
        df=df,
        batch_size=config.batch_size,
        test_size=1.0 - config.train_test_split,
        val_size=config.validation_split,
        num_workers=args.num_workers,
        random_seed=config.random_seed
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders.get('val')
    logger.info(f"Data loaded. Train batches: {len(train_loader)}")
    
    # Initialize model
    logger.info("Initializing DualEncoderModel...")
    model = DualEncoderModel()
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = TwoStageTrainer(model, config, device)
    
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
    
    args = parser.parse_args()
    
    main(args)
