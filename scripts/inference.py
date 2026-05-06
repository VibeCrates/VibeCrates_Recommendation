"""
Script for single inference - Make recommendations for a specific user
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.utils import load_model
import src.config as config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """
    TODO: Main inference script.
    
    Steps:
    1. Load trained model
    2. Get user data/profile
    3. Generate recommendations
    4. Print results
    """
    logger.info(f"Generating recommendations for user {args.user_id}...")
    
    try:
        logger.info(f"Loading model from {args.model_path}")
        model = load_model(args.model_path)
        
        logger.info(f"Getting recommendations (top {args.top_k})...")
        # TODO: Load user data and generate recommendations
        # recommendations = model.predict(user_data, top_k=args.top_k)
        
        logger.info("Recommendations:")
        # TODO: Print recommendations
        
        logger.info("Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate recommendations for a user")
    parser.add_argument("--user_id", type=int, required=True,
                       help="User ID to generate recommendations for")
    parser.add_argument("--model_path", type=str,
                       default=str(config.FINAL_MODEL_DIR / "model.pkl"),
                       help="Path to the trained model")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of recommendations to generate")
    
    args = parser.parse_args()
    main(args)
