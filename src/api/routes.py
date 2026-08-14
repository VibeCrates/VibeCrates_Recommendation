"""
API Routes
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from .dependencies import get_model_manager, get_manager_unchecked
from .schemas import (
    HealthCheckResponse,
    ItemInfoResponse,
    PingResponse,
    RecommendationRequest,
    RecommendationResponse,
)

router = APIRouter(prefix="/api/v1", tags=["recommendations"])


@router.get("/ping", response_model=PingResponse)
async def ping(
    n: int | None = Query(default=None, description="돌려받고 싶은 숫자. 생략하면 기본값 0"),
) -> PingResponse:
    """백엔드 ↔ 추천 서버 통신 확인용.

    **모델·인덱스에 의존하지 않는다.** 통신 경로(네트워크, 포트, 프록시, 컨테이너)만
    확인하는 것이 목적이므로, 모델이 없어도 200을 돌려줘야 실패 지점이 분리된다.
    n을 실어 보내면 그대로 돌려주므로 요청 본문이 실제로 전달됐는지도 함께 확인된다.
    """
    manager = get_manager_unchecked()
    return PingResponse(
        value=n if n is not None else 0,
        received=n,
        server_time=datetime.now(timezone.utc).isoformat(),
        model_loaded=manager.is_model_ready(),
    )


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
