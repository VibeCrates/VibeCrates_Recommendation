"""
API Routes
"""
from fastapi import APIRouter, Depends, HTTPException

from .dependencies import get_model_manager
from .schemas import (
    HealthCheckResponse,
    ItemInfoResponse,
    RecommendationRequest,
    RecommendationResponse,
)

router = APIRouter(prefix="/api/v1", tags=["recommendations"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(manager=Depends(get_model_manager)) -> HealthCheckResponse:
    return HealthCheckResponse(
        status="healthy",
        model_loaded=manager.is_model_ready(),
        index_built={domain: (domain in manager.indexes) for domain in ("movie", "music", "book")},
    )


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(
    request: RecommendationRequest,
    manager=Depends(get_model_manager),
) -> RecommendationResponse:
    """자연어 쿼리 → 도메인 아이템 top-K 추천."""
    if not manager.is_model_ready():
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
    if request.domain and request.domain not in manager.indexes:
        raise HTTPException(status_code=503, detail=f"{request.domain} 인덱스가 준비되지 않았습니다.")
    if not manager.indexes:
        raise HTTPException(status_code=503, detail="아직 준비된 인덱스가 없습니다.")

    results = manager.search(request.query, request.top_k, domain=request.domain)
    return RecommendationResponse(query=request.query, domain=request.domain, results=results)


@router.get("/item/{domain}/{item_id}", response_model=ItemInfoResponse)
async def get_item_info(
    domain: str,
    item_id: str,
    manager=Depends(get_model_manager),
) -> ItemInfoResponse:
    """도메인 아이템 메타데이터 조회."""
    if domain not in ("movie", "music", "book"):
        raise HTTPException(status_code=400, detail=f"지원하지 않는 도메인: {domain}")
    info = manager.get_item_info(domain, item_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"아이템을 찾을 수 없습니다: {item_id}")
    return ItemInfoResponse(item_id=item_id, domain=domain, info=info)
