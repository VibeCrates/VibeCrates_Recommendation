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

from src.data.dataset import MultiModalDataset, collate_fn
from src.data.meta import build_meta_df
from src.data.preprocessing import DOMAIN_CONFIG, prepare_domain_df
from src.models.recommender import DualEncoderModel
from torch.utils.data import DataLoader

MODEL_PATH = os.getenv("MODEL_PATH", "models/trained_model.pt")
IMAGE_DIR  = os.getenv("IMAGE_DIR",  "data/images")

INDEX_DIR = os.getenv("INDEX_DIR", "indexes")


def build_and_save(domain: str, model: DualEncoderModel, device: torch.device, batch_size: int,
                   index_dir: str = INDEX_DIR, image_embeddings=None) -> None:
    """image_embeddings를 반드시 학습 때와 같은 것으로 넘겨야 한다.

    넘기지 않으면 MultiModalDataset이 PIL 경로로 떨어져 **학습과 다른 이미지**를 쓴다.
    로컬 파일이 없는 항목은 학습 시 0 벡터였는데(dataset.py), 인덱싱 시에는 image_path에
    남아 있던 http URL로 실제 이미지를 내려받아 인코딩한다 — music 13,290건(33%),
    book 3%, movie 2.2%가 학습·검색에서 서로 다른 시각 신호를 갖게 된다.
    부작용으로 느리기도 하다: 8/7 실행에서 music 40K가 2시간 44분 걸렸다(다운로드 때문에,
    book 110K가 39분인 것과 대비된다).
    """
    cfg = DOMAIN_CONFIG[domain]
    df = pd.read_csv(cfg["csv"], low_memory=False)
    std_df = prepare_domain_df(domain, df, image_base_dir=IMAGE_DIR)

    print(f"[{domain}] {len(std_df):,} items → inferencing...")

    dataset = MultiModalDataset(
        content_texts=std_df["content_text"].tolist(),
        image_paths=std_df["image_path"].tolist(),
        queries=std_df["query"].fillna("").tolist(),
        image_embeddings=image_embeddings,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            z, _, _ = model.encode_content(batch["content_text"], batch["content_image"])
            all_embeddings.append(z.cpu())

    z_contents_n = F.normalize(torch.cat(all_embeddings), p=2, dim=1)
    meta_df = build_meta_df(domain, df, std_df)

    os.makedirs(index_dir, exist_ok=True)
    emb_path  = os.path.join(index_dir, f"{domain}_embeddings.pt")
    meta_path = os.path.join(index_dir, f"{domain}_meta.parquet")

    torch.save(z_contents_n, emb_path)
    meta_df.to_parquet(meta_path, index=False)

    print(f"[{domain}] Saved → {emb_path}  {z_contents_n.shape}")
    print(f"[{domain}] Saved → {meta_path}  ({len(meta_df):,} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=["movie", "music", "book"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--image-embeddings", default=None,
                        help="학습에 쓴 것과 같은 .pt. 생략하면 인덱스가 학습과 다른 "
                             "이미지를 쓰게 된다(build_and_save 주석 참조)")
    parser.add_argument("--index-dir", default=INDEX_DIR)
    parser.add_argument("--query-lora", action="store_true",
                        help="--query-lora로 학습한 체크포인트를 읽을 때 필요 (state_dict 키가 다르다)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_path} on {device}")
    model = DualEncoderModel(query_lora=args.query_lora)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    image_embeddings = None
    if args.image_embeddings:
        image_embeddings = torch.load(args.image_embeddings, map_location="cpu", weights_only=False)
        print(f"Loaded {len(image_embeddings):,} image embeddings")
    else:
        print("[warn] --image-embeddings 없음 — 인덱스가 학습과 다른 이미지를 쓴다")

    for domain in args.domains:
        build_and_save(domain, model, device, args.batch_size,
                       index_dir=args.index_dir, image_embeddings=image_embeddings)

    print("Done.")


if __name__ == "__main__":
    main()
