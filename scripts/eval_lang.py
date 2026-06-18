"""
한국어 vs 영어 쿼리 품질 비교 실험.

4가지 스타일 × 4쌍 × 2언어 = 32 쿼리
각 쿼리 × 4 도메인(movie/music/book/all) × top-5 = 최대 640행

출력: experiments/eval_lang_20260618.csv
"""
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
os.chdir(_root)

import pandas as pd
import torch
import torch.nn.functional as F

from src.models.recommender import DualEncoderModel

MODEL_PATH = "models/trained_model.pt"
INDEX_DIR  = "indexes"
OUT_DIR    = "experiments"
OUT_CSV    = f"{OUT_DIR}/eval_lang_20260618.csv"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────────────────────
# 쿼리 정의
# style: philosophical / poet / atmosphere / direct
# 각 스타일 4쌍, 한영 의미 동일
# ──────────────────────────────────────────────────────────────────────────────
QUERIES = [
    # ── 철학 페르소나 (Philosophical) ─────────────────────────────────────────
    ("P", 1, "ko", "인간이란 무엇인가"),
    ("P", 1, "en", "What is a human being"),
    ("P", 2, "ko", "삶의 의미를 찾는 여정"),
    ("P", 2, "en", "A journey to find the meaning of life"),
    ("P", 3, "ko", "자유와 책임 사이에서"),
    ("P", 3, "en", "Between freedom and responsibility"),
    ("P", 4, "ko", "고독과 존재에 대한 성찰"),
    ("P", 4, "en", "Reflection on solitude and existence"),

    # ── 시인 페르소나 (Poet) ──────────────────────────────────────────────────
    ("T", 1, "ko", "사랑의 이름으로"),
    ("T", 1, "en", "In the name of love"),
    ("T", 2, "ko", "봄날의 설레임처럼"),
    ("T", 2, "en", "Like the excitement of a spring day"),
    ("T", 3, "ko", "이별 후에 남는 것들"),
    ("T", 3, "en", "What remains after farewell"),
    ("T", 4, "ko", "달빛 아래 속삭이는 목소리"),
    ("T", 4, "en", "A voice whispering under moonlight"),

    # ── 공간 페르소나 (Atmosphere) ────────────────────────────────────────────
    ("A", 1, "ko", "햇살이 드는 카페"),
    ("A", 1, "en", "A sunlit cafe"),
    ("A", 2, "ko", "빗소리 들리는 조용한 오후"),
    ("A", 2, "en", "A quiet afternoon with the sound of rain"),
    ("A", 3, "ko", "한겨울 따뜻한 벽난로 앞에서"),
    ("A", 3, "en", "In front of a warm fireplace in midwinter"),
    ("A", 4, "ko", "새벽 도심의 텅 빈 거리"),
    ("A", 4, "en", "Empty city streets at dawn"),

    # ── 직접 특성 (Direct) ────────────────────────────────────────────────────
    ("D", 1, "ko", "우주를 배경으로 한 액션 영화"),
    ("D", 1, "en", "Action movie set in space"),
    ("D", 2, "ko", "재즈 피아노가 흐르는 감성적인 음악"),
    ("D", 2, "en", "Soulful music featuring jazz piano"),
    ("D", 3, "ko", "반전이 있는 미스터리 스릴러 소설"),
    ("D", 3, "en", "Mystery thriller novel with a twist ending"),
    ("D", 4, "ko", "두 사람의 로맨스를 다룬 영화"),
    ("D", 4, "en", "A film about a romance between two people"),
]

STYLE_NAMES = {"P": "philosophical", "T": "poet", "A": "atmosphere", "D": "direct"}
DOMAINS = ["movie", "music", "book", "all"]
TOP_K = 5


def load_model(path: str) -> DualEncoderModel:
    print(f"모델 로딩: {path}", flush=True)
    model = DualEncoderModel()
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def load_indexes(index_dir: str) -> dict:
    indexes = {}
    for domain in ("movie", "music", "book"):
        emb_path  = f"{index_dir}/{domain}_embeddings.pt"
        meta_path = f"{index_dir}/{domain}_meta.parquet"
        if os.path.exists(emb_path) and os.path.exists(meta_path):
            z = torch.load(emb_path, map_location="cpu", weights_only=False)
            meta = pd.read_parquet(meta_path)
            meta["_domain"] = domain
            indexes[domain] = (z, meta)
            print(f"  [{domain}] {z.shape[0]:,}개 아이템")
    return indexes


@torch.no_grad()
def search(model, indexes, query: str, domain_filter: str | None, top_k: int) -> list[dict]:
    z_q = model.encode_query([query])
    z_q_n = F.normalize(z_q, p=2, dim=1).cpu()

    target = {domain_filter: indexes[domain_filter]} if domain_filter else indexes

    all_scores, all_metas = [], []
    for d, (z_n, meta) in target.items():
        scores = (z_q_n @ z_n.T).squeeze(0)
        all_scores.append(scores)
        all_metas.append(meta)

    merged_scores = torch.cat(all_scores)
    merged_meta   = pd.concat(all_metas, ignore_index=True)

    k = min(top_k, len(merged_scores))
    top_idx = merged_scores.topk(k).indices.tolist()

    results = []
    for rank, idx in enumerate(top_idx, 1):
        row = merged_meta.iloc[idx]
        extra = ""
        for key in ("artist", "director", "author"):
            if key in row and pd.notna(row.get(key)):
                extra = str(row[key])
                break
        results.append({
            "result_domain": str(row["_domain"]),
            "item_id":       str(row.get("item_id", "")),
            "title":         str(row.get("title", "")),
            "extra":         extra,
            "score":         float(merged_scores[idx]),
            "rank":          rank,
        })
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model   = load_model(MODEL_PATH)
    indexes = load_indexes(INDEX_DIR)
    print(f"준비 완료. (device: {DEVICE})\n")

    rows = []
    total = len(QUERIES) * len(DOMAINS)
    done  = 0

    for style_code, pair_id, lang, query in QUERIES:
        for domain_filter in DOMAINS:
            df_arg = None if domain_filter == "all" else domain_filter
            if df_arg and df_arg not in indexes:
                continue
            results = search(model, indexes, query, df_arg, TOP_K)
            for r in results:
                rows.append({
                    "query_id":     f"{style_code}{pair_id}_{lang.upper()}",
                    "style":        STYLE_NAMES[style_code],
                    "pair_id":      pair_id,
                    "lang":         lang,
                    "query":        query,
                    "domain_filter": domain_filter,
                    **r,
                })
            done += 1
            if done % 20 == 0:
                print(f"  진행: {done}/{total}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUT_CSV}  ({len(df):,}행)\n")

    # ── 간단 자동 통계 ────────────────────────────────────────────────────────
    print("=== 언어별 평균 Top-5 Score ===")
    summary = (
        df.groupby(["lang", "domain_filter"])["score"]
        .mean()
        .round(4)
        .unstack("domain_filter")
    )
    print(summary.to_string())

    print("\n=== 스타일 × 언어별 평균 Score ===")
    style_summary = (
        df.groupby(["style", "lang"])["score"]
        .mean()
        .round(4)
        .unstack("lang")
    )
    print(style_summary.to_string())


if __name__ == "__main__":
    main()
