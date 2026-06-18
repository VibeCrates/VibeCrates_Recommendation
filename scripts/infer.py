"""
Quick inference script — 쿼리 입력 시 추천 아이템 출력.

Usage:
  python scripts/infer.py "우울한 날 기분 전환되는 신나는 음악"
  python scripts/infer.py "우주를 배경으로 한 SF 액션 영화" --domain movie --top-k 5
  python scripts/infer.py --interactive
"""
import argparse
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import pandas as pd
import torch
import torch.nn.functional as F

from src.models.recommender import DualEncoderModel

MODEL_PATH = os.path.join(_root, "models", "trained_model.pt")
INDEX_DIR  = os.path.join(_root, "indexes")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path: str) -> DualEncoderModel:
    print(f"모델 로딩: {path}", flush=True)
    model = DualEncoderModel()
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def load_indexes(index_dir: str) -> dict:
    indexes = {}
    for domain in ("movie", "music", "book"):
        emb_path  = os.path.join(index_dir, f"{domain}_embeddings.pt")
        meta_path = os.path.join(index_dir, f"{domain}_meta.parquet")
        if os.path.exists(emb_path) and os.path.exists(meta_path):
            z = torch.load(emb_path, map_location="cpu")
            meta = pd.read_parquet(meta_path)
            indexes[domain] = (z, meta)
            print(f"  [{domain}] {z.shape[0]:,}개 아이템 로드")
        else:
            print(f"  [{domain}] 인덱스 없음 — 건너뜀")
    return indexes


@torch.no_grad()
def search(model: DualEncoderModel, indexes: dict, query: str, top_k: int, domain: str | None):
    z_query = model.encode_query([query])
    z_query_n = F.normalize(z_query, p=2, dim=1).cpu()

    target = {domain: indexes[domain]} if domain and domain in indexes else indexes

    all_scores, all_metas = [], []
    for d, (z_contents_n, meta_df) in target.items():
        scores = (z_query_n @ z_contents_n.T).squeeze(0)
        all_scores.append(scores)
        meta_df = meta_df.copy()
        meta_df["_domain"] = d
        all_metas.append(meta_df)

    merged_scores = torch.cat(all_scores)
    merged_meta   = pd.concat(all_metas, ignore_index=True)

    top_idx = merged_scores.topk(min(top_k, len(merged_scores))).indices.tolist()

    results = []
    for rank, idx in enumerate(top_idx, 1):
        row   = merged_meta.iloc[idx]
        score = float(merged_scores[idx])
        d     = row["_domain"]
        title = str(row.get("title", ""))

        extra_parts = []
        for key in ("artist", "director", "author"):
            if key in row and pd.notna(row[key]):
                extra_parts.append(str(row[key]))
                break

        extra_str = f" / {extra_parts[0]}" if extra_parts else ""
        print(f"  {rank:2}. [{d:5}] {title}{extra_str}  (score: {score:.4f})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--domain", choices=["movie", "music", "book"], default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--interactive", action="store_true", help="대화형 모드")
    args = parser.parse_args()

    model   = load_model(MODEL_PATH)
    indexes = load_indexes(INDEX_DIR)
    print(f"준비 완료. (device: {DEVICE})\n")

    def run(query: str, domain: str | None):
        print(f"\n쿼리: {query}")
        print("-" * 60)
        search(model, indexes, query, args.top_k, domain)

    if args.interactive:
        print("쿼리를 입력하세요. 종료: q / 도메인 접두어: movie: / music: / book:")
        while True:
            try:
                line = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line or line.lower() == "q":
                break
            domain = args.domain
            query  = line
            for d in ("movie", "music", "book"):
                if line.lower().startswith(f"{d}:"):
                    domain = d
                    query  = line[len(d) + 1:].strip()
                    break
            run(query, domain)
    elif args.query:
        run(args.query, args.domain)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
