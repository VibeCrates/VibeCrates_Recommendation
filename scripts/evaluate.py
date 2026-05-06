"""
Script to evaluate the trained recommendation model
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.models.utils import load_model
import src.config as config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """
    TODO: Main evaluation script.
    
    Steps:
    1. Load trained model
    2. Load test data
    3. Generate predictions
    4. Calculate evaluation metrics (Precision@K, Recall@K, NDCG, etc.)
    5. Print results
    """
    logger.info("Starting model evaluation...")
    
    try:
        logger.info(f"Loading model from {args.model_path}")
        model = load_model(args.model_path)
        
        logger.info("Loading test data...")
        data_loader = DataLoader(data_path=str(config.PROCESSED_DATA_DIR))
        # TODO: Load test data
        
        logger.info("Generating predictions...")
        # TODO: Generate predictions using model
        
        logger.info("Calculating metrics...")
        # TODO: Calculate evaluation metrics
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate recommendation model")
    parser.add_argument("--model_path", type=str, 
                       default=str(config.FINAL_MODEL_DIR / "model.pkl"),
                       help="Path to the trained model")
    
    args = parser.parse_args()
    main(args)
