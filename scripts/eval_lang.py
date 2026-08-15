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

MODEL_PATH = os.environ.get("EVAL_MODEL_PATH", "models/trained_model.pt")
INDEX_DIR  = os.environ.get("EVAL_INDEX_DIR", "indexes")
OUT_DIR    = "experiments"
# 실행 날짜로 파일명을 만든다. 이전에는 20260618로 하드코딩돼 있어 재실행하면 6월
# baseline(eval_lang_20260618.csv)을 덮어썼다 — 비교 대상 자체가 사라진다.
OUT_CSV    = os.environ.get(
    "EVAL_OUT_CSV", f"{OUT_DIR}/eval_lang_{__import__('datetime').date.today():%Y%m%d}.csv"
)
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

    # ─────────────────────────────────────────────────────────────────────────
    # 확장분 (pair_id 5~10, 2026-08-14 추가)
    # 기존 1~4는 6월 사람 라벨이 붙어 있는 고정 집합이므로 건드리지 않는다 —
    # baseline 비교는 그 부분집합으로만 해야 성립한다.
    # 확장 이유: 스타일당 4쌍 × top-5면 스타일당 80행뿐이라 표준오차가 ~0.1이고,
    # 실제로 stage2 손실 수정의 poet 개선(+0.138)이 1.5 SE에 그쳐 판정이 불가능했다.
    # 스타일당 10쌍 × top-10이면 스타일당 400행으로 SE가 절반 이하가 된다.
    # ─────────────────────────────────────────────────────────────────────────

    # 철학 (Philosophical)
    ("P", 5, "ko", "기억은 나를 얼마나 만드는가"),
    ("P", 5, "en", "How much of me is made of memory"),
    ("P", 6, "ko", "우연과 운명의 경계"),
    ("P", 6, "en", "The border between chance and fate"),
    ("P", 7, "ko", "타인을 이해한다는 것의 한계"),
    ("P", 7, "en", "The limits of understanding another person"),
    ("P", 8, "ko", "죽음을 앞에 둔 삶의 무게"),
    ("P", 8, "en", "The weight of a life facing its end"),
    ("P", 9, "ko", "옳음과 다정함 중 무엇을 택할까"),
    ("P", 9, "en", "Choosing between being right and being kind"),
    ("P", 10, "ko", "반복되는 일상 속의 의미"),
    ("P", 10, "en", "Meaning inside a repeating daily life"),

    # 시인 (Poet) — 추상·감각 이미지. 6월 최대 실패 지점이라 표본을 두텁게 둔다.
    ("T", 5, "ko", "식어가는 커피처럼 멀어지는 마음"),
    ("T", 5, "en", "A heart cooling like forgotten coffee"),
    ("T", 6, "ko", "오래된 편지에서 나는 냄새"),
    ("T", 6, "en", "The smell rising from an old letter"),
    ("T", 7, "ko", "유리창에 맺힌 겨울 숨결"),
    ("T", 7, "en", "Winter breath fogging a windowpane"),
    ("T", 8, "ko", "말하지 못한 채 지나간 계절"),
    ("T", 8, "en", "A season that passed without being spoken"),
    ("T", 9, "ko", "모래처럼 빠져나가는 시간"),
    ("T", 9, "en", "Time slipping away like sand"),
    ("T", 10, "ko", "불 꺼진 방에 남은 온기"),
    ("T", 10, "en", "Warmth left in a room after the lights go out"),

    # 공간 (Atmosphere)
    ("A", 5, "ko", "여름밤 열어둔 창가"),
    ("A", 5, "en", "An open window on a summer night"),
    ("A", 6, "ko", "눈 내리는 밤의 시골 기차역"),
    ("A", 6, "en", "A country train station on a snowy night"),
    ("A", 7, "ko", "늦은 밤 편의점의 형광등 불빛"),
    ("A", 7, "en", "Fluorescent glow of a late-night convenience store"),
    ("A", 8, "ko", "바닷가 낡은 모텔 방"),
    ("A", 8, "en", "A worn motel room by the sea"),
    ("A", 9, "ko", "장마철 눅눅한 지하 서점"),
    ("A", 9, "en", "A damp basement bookshop during the rainy season"),
    ("A", 10, "ko", "해질녘 고속도로 휴게소"),
    ("A", 10, "en", "A highway rest stop at sunset"),

    # 직접 (Direct)
    ("D", 5, "ko", "실화를 바탕으로 한 법정 드라마"),
    ("D", 5, "en", "A courtroom drama based on a true story"),
    ("D", 6, "ko", "기타 리프가 강렬한 록 음악"),
    ("D", 6, "en", "Rock music with a heavy guitar riff"),
    ("D", 7, "ko", "가족의 비밀을 다룬 소설"),
    ("D", 7, "en", "A novel about a family secret"),
    ("D", 8, "ko", "1980년대를 배경으로 한 청춘 영화"),
    ("D", 8, "en", "A coming-of-age film set in the 1980s"),
    ("D", 9, "ko", "차분한 피아노 연주곡"),
    ("D", 9, "en", "A calm solo piano piece"),
    ("D", 10, "ko", "우주 탐사를 다룬 논픽션"),
    ("D", 10, "en", "Nonfiction about space exploration"),
]

# 6월 baseline과 비교 가능한 부분집합. 확장분을 섞어 평균 내면 baseline 대비 수치가
# 무의미해지므로, 리포트에서 이 집합을 따로 볼 수 있도록 pair_id로 구분한다.
BASELINE_PAIR_IDS = {1, 2, 3, 4}

STYLE_NAMES = {"P": "philosophical", "T": "poet", "A": "atmosphere", "D": "direct"}
DOMAINS = ["movie", "music", "book", "all"]
TOP_K = 10   # 5 → 10 (2026-08-14): 표본을 늘려 표준오차를 낮춘다


# --query-lora로 학습한 체크포인트는 state_dict 키가 다르므로 같은 구조로 만들어야 한다.
QUERY_LORA = os.environ.get("EVAL_QUERY_LORA", "0") == "1"


def load_model(path: str) -> DualEncoderModel:
    print(f"모델 로딩: {path} (query_lora={QUERY_LORA})", flush=True)
    model = DualEncoderModel(query_lora=QUERY_LORA)
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
