"""
Pre-extracts CLIP vision encoder outputs for all images in train_dataset.csv.
Saves {image_path: np.array(1024)} to data/image_embeddings.pt.

Since the CLIP vision encoder is frozen during training, this only needs to run once.
The MLP head (trainable) is applied at training time.

Usage:
  python scripts/extract_image_embeddings.py
  python scripts/extract_image_embeddings.py --batch-size 128 --output data/image_embeddings.pt
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModel

MODEL_NAME = "openai/clip-vit-large-patch14"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/train_dataset.csv")
    parser.add_argument("--output", default="data/image_embeddings.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading CLIP: {MODEL_NAME}")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    vision_encoder = CLIPVisionModel.from_pretrained(MODEL_NAME).to(device).eval()

    df = pd.read_csv(args.dataset)
    # Deduplicate paths (same image can appear in train/val/test split)
    unique_paths = list(dict.fromkeys(df["image_path"].tolist()))
    print(f"{len(unique_paths):,} unique images")

    embeddings = {}
    failed = 0

    for i in tqdm(range(0, len(unique_paths), args.batch_size), desc="extracting"):
        batch_paths = unique_paths[i : i + args.batch_size]
        images, valid_paths = [], []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_paths.append(path)
            except Exception:
                failed += 1

        if not images:
            continue

        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt").to(device)
            features = vision_encoder(**inputs).pooler_output.cpu().numpy()  # (N, 1024)

        for path, feat in zip(valid_paths, features):
            embeddings[path] = feat

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(embeddings, args.output)
    print(f"saved {len(embeddings):,} embeddings → {args.output}")
    if failed:
        print(f"WARNING: {failed} images failed to load and were skipped")


if __name__ == "__main__":
    main()
