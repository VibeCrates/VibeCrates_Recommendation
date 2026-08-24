"""
Dependency injection for FastAPI — 모델 로드 및 아이템 인덱스 관리.
"""
import hashlib
import logging
import os
from functools import lru_cache
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F

from .schemas import RecommendationItem

# 무거운 것(DualEncoderModel → sentence-transformers/transformers/peft)은 여기서 import
# 하지 않고 실제로 모델을 쓸 때 끌어온다. /api/v1/ping은 통신 경로만 확인하는 엔드포인트라
# 모델 스택이 설치돼 있지 않아도 떠야 하는데, 모듈 최상단에서 import 하면 routes.py →
# dependencies.py 체인을 타고 앱 자체가 뜨지 않는다.

logger = logging.getLogger(__name__)

MODEL_PATH  = os.getenv("MODEL_PATH",  "models/trained_model.pt")
IMAGE_DIR   = os.getenv("IMAGE_DIR",   "data/images")
INDEX_DIR   = os.getenv("INDEX_DIR",   "indexes")
DEVICE      = os.getenv("DEVICE",      "cuda" if torch.cuda.is_available() else "cpu")

# 백엔드 연동 시험용 가짜 벡터 모드.
#   모델이 아직 이 컴퓨터에 없어도 백엔드가 **응답 모양**(차원·정규화·키 이름·JSON 직렬화)을
#   미리 맞춰볼 수 있어야 한다. 통신 확인을 모델 적재와 분리한 /ping과 같은 취지다.
#   진짜 벡터와 섞이면 조용히 틀린 검색 결과가 나오므로, model_version에 FAKE를 박아
#   백엔드가 응답만 보고도 구별할 수 있게 한다.
FAKE_VECTOR = os.getenv("FAKE_VECTOR", "0").lower() not in ("0", "", "false", "no")
EMBED_DIM   = int(os.getenv("EMBED_DIM", "768"))


class ModelManager:
    """
    모델과 도메인별 아이템 임베딩 인덱스를 관리.

    - model: DualEncoderModel (추론 전용)
    - indexes[domain]: (z_contents_n, meta_df) 쌍
      z_contents_n : (N, 768) L2-정규화된 아이템 임베딩 CPU tensor
      meta_df      : 아이템 메타데이터 DataFrame (item_id, title, extra)
    """

    def __init__(self):
        self.model: Optional[DualEncoderModel] = None
        self.device = torch.device(DEVICE)
        self.indexes: dict[str, tuple[torch.Tensor, pd.DataFrame]] = {}
        # 기동 시 실패한 것과 그 이유. 앱은 실패해도 뜨기 때문에(/ping은 모델과 무관하게
        # 응답해야 한다) 실패 사실이 어딘가 남아 있지 않으면 "떴는데 추천만 503"이라는
        # 헷갈리는 상태가 된다. /api/v1/health가 이 값을 그대로 내보낸다.
        self.load_errors: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 모델
    # ------------------------------------------------------------------

    def load_model(self, model_path: str = MODEL_PATH) -> None:
        from src.models.recommender import DualEncoderModel

        logger.info(f"Loading model from {model_path}")
        model = DualEncoderModel()
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model
        logger.info("Model loaded.")

    def is_model_ready(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    # 아이템 인덱스
    # ------------------------------------------------------------------

    def load_index(self, domain: str, index_dir: str = INDEX_DIR) -> bool:
        """저장된 임베딩 파일이 있으면 로드. 성공 시 True 반환."""
        emb_path  = os.path.join(index_dir, f"{domain}_embeddings.pt")
        meta_path = os.path.join(index_dir, f"{domain}_meta.parquet")
        if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
            return False
        z_contents_n = torch.load(emb_path, map_location="cpu")
        meta_df = pd.read_parquet(meta_path)
        meta_df["domain"] = domain
        self.indexes[domain] = (z_contents_n, meta_df)
        logger.info(f"[{domain}] Index loaded from disk: {z_contents_n.shape}")
        return True

    @torch.no_grad()
    def build_index(self, domain: str, batch_size: int = 64) -> None:
        """
        도메인 CSV를 읽어 전체 아이템 임베딩을 생성하고 메모리에 캐싱.
        모델이 로드된 후 호출해야 함.
        """
        if not self.is_model_ready():
            raise RuntimeError("모델을 먼저 로드하세요.")

        from src.data.meta import build_meta_df
        from src.data.preprocessing import DOMAIN_CONFIG, prepare_domain_df

        cfg = DOMAIN_CONFIG[domain]
        df = pd.read_csv(cfg["csv"], low_memory=False)
        std_df = prepare_domain_df(domain, df, image_base_dir=IMAGE_DIR)

        logger.info(f"[{domain}] Building index for {len(std_df):,} items...")

        from torch.utils.data import DataLoader as TorchDataLoader
        from src.data.dataset import MultiModalDataset, collate_fn

        dataset = MultiModalDataset(
            content_texts=std_df["content_text"].tolist(),
            image_paths=std_df["image_path"].tolist(),
            queries=std_df["query"].fillna("").tolist(),
        )
        loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        all_embeddings = []
        self.model.eval()
        for batch in loader:
            z, _, _ = self.model.encode_content(batch["content_text"], batch["content_image"])
            all_embeddings.append(z.cpu())

        z_contents = torch.cat(all_embeddings)
        z_contents_n = F.normalize(z_contents, p=2, dim=1)

        # 메타 정보 구성 (도메인별)
        meta_df = build_meta_df(domain, df, std_df)

        self.indexes[domain] = (z_contents_n, meta_df)
        logger.info(f"[{domain}] Index built: {z_contents_n.shape}")

    # ------------------------------------------------------------------
    # 검색
    # ------------------------------------------------------------------

    @torch.no_grad()
    def search(self, query: str, top_k: int, domain: str | None = None) -> list[RecommendationItem]:
        """
        domain=None 이면 준비된 모든 도메인을 합쳐 통합 검색.
        domain 지정 시 해당 도메인만 검색.
        """
        target = {domain: self.indexes[domain]} if domain else self.indexes

        self.model.eval()
        z_query = self.model.encode_query([query])
        z_query_n = F.normalize(z_query, p=2, dim=1).cpu()

        all_scores, all_meta = [], []
        for d, (z_contents_n, meta_df) in target.items():
            scores = (z_query_n @ z_contents_n.T).squeeze(0)
            all_scores.append(scores)
            all_meta.append(meta_df)

        merged_scores = torch.cat(all_scores)
        merged_meta   = pd.concat(all_meta, ignore_index=True)

        top_idx = merged_scores.topk(top_k).indices.tolist()
        results = []
        for idx in top_idx:
            row = merged_meta.iloc[idx]
            results.append(RecommendationItem(
                item_id=str(row["item_id"]),
                domain=str(row["domain"]),
                score=float(merged_scores[idx]),
                title=str(row["title"]),
                extra=row.get("extra"),
            ))
        return results

    # ------------------------------------------------------------------
    # 쿼리 임베딩 (백엔드가 자기 쪽에서 검색하는 경로)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_queries(self, queries: list[str], normalize: bool = True) -> torch.Tensor:
        """자연어 쿼리 → z_query (B, 768).

        search()가 내부에서 쓰는 것과 **같은 경로**를 타야 백엔드가 받은 벡터로
        우리 아이템 임베딩을 검색했을 때 결과가 일치한다. 그래서 encode_query를
        그대로 쓰고 정규화 여부만 파라미터로 연다. 기본값이 True인 이유는
        search()가 L2 정규화 후 내적하기 때문이다 — 내적이 곧 코사인 유사도가 된다.
        """
        if FAKE_VECTOR and not self.is_model_ready():
            return torch.stack([self._fake_vector(q, normalize) for q in queries])

        self.model.eval()
        z = self.model.encode_query(queries)
        if normalize:
            z = F.normalize(z, p=2, dim=1)
        return z.cpu()

    @staticmethod
    def _fake_vector(keyword: str, normalize: bool = True) -> torch.Tensor:
        """검색어로 시드를 고정한 난수 벡터. 의미는 전혀 없고 **모양만** 진짜와 같다.

        시드를 고정하는 이유: 같은 검색어에 항상 같은 벡터가 나와야 백엔드가 캐싱이나
        재현성을 시험할 수 있고, 매번 달라지면 "내가 뭘 잘못 보냈나"와 구별되지 않는다.
        """
        seed = int(hashlib.sha256(keyword.encode("utf-8")).hexdigest()[:16], 16) % (2**63)
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(EMBED_DIM, generator=g)
        if normalize:
            z = F.normalize(z, p=2, dim=0)
        return z

    def can_embed(self) -> bool:
        """임베딩 응답을 낼 수 있는가. 가짜 모드에서는 모델 없이도 낼 수 있다."""
        return self.is_model_ready() or FAKE_VECTOR

    def model_version(self) -> str:
        """쿼리 벡터와 아이템 임베딩이 같은 체크포인트에서 나왔는지 대조하는 값.

        둘이 어긋나면 검색은 오류 없이 성공하고 결과만 엉뚱해진다 — 조용히 틀리는
        종류의 사고라 백엔드가 스스로 감지할 수단이 있어야 한다.
        """
        if not self.is_model_ready():
            return "FAKE-no-model" if FAKE_VECTOR else "unknown"
        try:
            return f"{os.path.basename(MODEL_PATH)}@{int(os.path.getmtime(MODEL_PATH))}"
        except OSError:
            return "unknown"

    # ------------------------------------------------------------------
    # 아이템 메타 조회
    # ------------------------------------------------------------------

    def get_item_info(self, domain: str, item_id: str) -> Optional[dict]:
        if domain not in self.indexes:
            return None
        _, meta_df = self.indexes[domain]
        rows = meta_df[meta_df["item_id"].astype(str) == item_id]
        if rows.empty:
            return None
        row = rows.iloc[0]
        return {"title": row["title"], **(row.get("extra") or {})}


# ---------------------------------------------------------------------------
# FastAPI 의존성
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_model_manager() -> ModelManager:
    return _manager


def get_manager_unchecked() -> ModelManager:
    """모델 준비 여부와 무관하게 매니저를 돌려준다 (/ping 전용).

    get_model_manager와 지금은 같지만, 나중에 전자에 "모델 없으면 503" 같은 검사를
    붙이더라도 통신 확인 경로는 그 영향을 받지 않아야 하므로 분리해 둔다.
    """
    return _manager


_manager = ModelManager()


def _hint_for(exc: Exception) -> str:
    """자주 겪은 실패에 대해 다음에 할 일을 한 줄로 알려준다.

    예외 메시지만으로는 무엇을 해야 하는지 알기 어려운 경우가 반복됐다. pyarrow 결손은
    8/7 서버에서 같은 원인으로 3일을 멈췄고, venv 밖 파이썬 문제는 8/18에 겪었다.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if "pyarrow" in text or "fastparquet" in text:
        return "→ parquet 엔진이 없습니다. `./venv/bin/pip install -r requirements-serve.txt`"
    if "no module named" in text:
        return "→ venv 밖의 파이썬으로 떴을 수 있습니다. `./scripts/serve.sh`로 실행하세요."
    if "unrecognized image processor" in text:
        return "→ 백로그 D11. QueryBlock의 CLIPProcessor를 CLIPTokenizer로 바꿔야 합니다."
    if isinstance(exc, FileNotFoundError):
        return f"→ 경로 확인: MODEL_PATH={MODEL_PATH} / INDEX_DIR={INDEX_DIR}"
    return "→ 위 예외를 그대로 확인하세요."


def _log_failure(what: str, exc: Exception, consequence: str) -> None:
    """실패를 눈에 띄게 남긴다. 조용히 지나가면 원인을 찾는 데 며칠이 걸린다."""
    logger.error("=" * 72)
    logger.error(f"!! {what} 실패 — {consequence}")
    logger.error(f"   {type(exc).__name__}: {exc}")
    logger.error(f"   {_hint_for(exc)}")
    logger.error("=" * 72)


async def initialize_dependencies() -> None:
    """모델과 인덱스를 적재한다. 실패해도 앱은 뜬다 — 다만 조용히 넘어가지는 않는다.

    앱이 뜨는 것은 의도된 동작이다. /ping은 모델 스택과 무관하게 응답해야 통신 경로만
    따로 확인할 수 있다. 문제는 예외를 삼키기만 하던 이전 구현이었다. 실패해도 앱이
    뜨므로 백엔드 쪽에서는 "서버는 사는데 추천만 503"으로 보였고, 원인이 로그 한 줄로만
    남아 묻혔다. 이제 실패를 배너로 찍고 manager.load_errors에 남겨 /health로 내보낸다.
    """
    logger.info("Initializing dependencies...")
    _manager.load_errors.clear()

    try:
        _manager.load_model()
    except Exception as e:
        _manager.load_errors["model"] = f"{type(e).__name__}: {e}"
        _log_failure("모델 적재", e, "추천·임베딩이 모두 503이 됩니다")

    for domain in ("movie", "music", "book"):
        try:
            if not _manager.load_index(domain):
                msg = f"인덱스 파일 없음 ({INDEX_DIR}/{domain}_embeddings.pt, _meta.parquet)"
                _manager.load_errors[domain] = msg
                logger.warning(f"[{domain}] {msg} — `python scripts/build_index.py` 후 다시 띄우세요.")
        except Exception as e:
            _manager.load_errors[domain] = f"{type(e).__name__}: {e}"
            _log_failure(f"[{domain}] 인덱스 적재", e, "이 도메인 추천이 503이 됩니다")

    if _manager.load_errors:
        logger.error(
            "기동은 됐지만 실패한 것이 있습니다: %s. 자세한 내용은 GET /api/v1/health 의 errors.",
            ", ".join(_manager.load_errors),
        )
    else:
        logger.info("모델·인덱스 적재 완료.")


async def cleanup_dependencies() -> None:
    logger.info("Cleaning up...")
    _manager.indexes.clear()
    _manager.model = None
