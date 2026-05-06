"""
Script to train the recommendation model
"""
import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.models.recommender import CollaborativeFilteringRecommender
from src.training.trainer import train_model
from src.training.config import TrainingConfig
import src.config as config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """
    TODO: Main training script.
    
    Steps:
    1. Load data
    2. Preprocess data
    3. Initialize model
    4. Train model
    5. Save trained model
    6. Evaluate on test set
    """
    logger.info("Starting model training...")
    
    try:
        # Load training config
        training_config = TrainingConfig(
            model_type=args.model_type,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
        )
        
        logger.info("Loading data...")
        data_loader = DataLoader(data_path=str(config.PROCESSED_DATA_DIR))
        # TODO: Load data using data_loader
        
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor()
        # TODO: Preprocess data
        
        logger.info("Initializing model...")
        model = CollaborativeFilteringRecommender()
        
        logger.info("Training model...")
        # TODO: Train model using training_config
        
        logger.info("Saving model...")
        # TODO: Save trained model to config.FINAL_MODEL_DIR
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train recommendation model")
    parser.add_argument("--model_type", type=str, default="collaborative_filtering",
                       help="Type of model to train")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    
    args = parser.parse_args()
    main(args)
