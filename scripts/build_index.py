"""
학습 완료 후 한 번 실행해서 도메인별 아이템 임베딩을 indexes/ 에 저장.

Usage:
    python scripts/build_index.py [--domains movie music book] [--batch-size 64]
"""
import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.api.dependencies import MODEL_PATH, IMAGE_DIR, _build_meta_df
from src.data.dataset import MultiModalDataset, collate_fn
from src.data.preprocessing import DOMAIN_CONFIG, prepare_domain_df
from src.models.recommender import DualEncoderModel
from torch.utils.data import DataLoader

INDEX_DIR = os.getenv("INDEX_DIR", "indexes")


def build_and_save(domain: str, model: DualEncoderModel, device: torch.device, batch_size: int) -> None:
    cfg = DOMAIN_CONFIG[domain]
    df = pd.read_csv(cfg["csv"], low_memory=False)
    std_df = prepare_domain_df(domain, df, image_base_dir=IMAGE_DIR)

    print(f"[{domain}] {len(std_df):,} items → inferencing...")

    dataset = MultiModalDataset(
        content_texts=std_df["content_text"].tolist(),
        image_paths=std_df["image_path"].tolist(),
        queries=std_df["query"].fillna("").tolist(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            z, _, _ = model.encode_content(batch["content_text"], batch["content_image"])
            all_embeddings.append(z.cpu())

    z_contents_n = F.normalize(torch.cat(all_embeddings), p=2, dim=1)
    meta_df = _build_meta_df(domain, df, std_df)

    os.makedirs(INDEX_DIR, exist_ok=True)
    emb_path  = os.path.join(INDEX_DIR, f"{domain}_embeddings.pt")
    meta_path = os.path.join(INDEX_DIR, f"{domain}_meta.parquet")

    torch.save(z_contents_n, emb_path)
    meta_df.to_parquet(meta_path, index=False)

    print(f"[{domain}] Saved → {emb_path}  {z_contents_n.shape}")
    print(f"[{domain}] Saved → {meta_path}  ({len(meta_df):,} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=["movie", "music", "book"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-path", default=MODEL_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_path} on {device}")
    model = DualEncoderModel()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    for domain in args.domains:
        build_and_save(domain, model, device, args.batch_size)

    print("Done.")


if __name__ == "__main__":
    main()
