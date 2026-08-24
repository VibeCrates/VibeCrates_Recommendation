"""
API Routes
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from .dependencies import get_model_manager, get_manager_unchecked
from .schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    HealthCheckResponse,
    ItemInfoResponse,
    PingResponse,
    RecommendationRequest,
    RecommendationResponse,
    TextVectorRequest,
    TextVectorResponse,
    VectorSearchResponse,
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


@router.post("/embeddings", response_model=EmbeddingResponse)
async def embeddings(
    request: EmbeddingRequest,
    manager=Depends(get_model_manager),
) -> EmbeddingResponse:
    """자연어 쿼리를 QueryBlock에 통과시킨 벡터를 돌려준다.

    /recommend와 달리 **검색을 하지 않는다.** 백엔드가 자기 벡터 DB에서 직접 검색하는
    구조를 위한 엔드포인트이므로, 아이템 임베딩(indexes/*.pt)을 같은 체크포인트로 미리
    받아 두어야 의미가 있다. 응답의 model_version이 그 대조용이다.
    """
    if not manager.can_embed():
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    queries = request.values
    if not queries:
        raise HTTPException(status_code=422, detail="query(또는 text)가 비어 있습니다.")

    z = manager.encode_queries(queries, normalize=request.normalize)
    return EmbeddingResponse(
        dim=z.shape[1],
        normalized=request.normalize,
        model_version=manager.model_version(),
        vectors=z.tolist(),
    )


@router.get("/search/vector", response_model=VectorSearchResponse)
async def search_vector(
    keyword: str = Query(..., min_length=1, description="자연어 검색어"),
    normalize: bool = Query(default=True, description="L2 정규화 여부"),
    manager=Depends(get_model_manager),
) -> VectorSearchResponse:
    """검색어 하나를 쿼리 벡터로 바꿔 돌려준다 (백엔드 요청 형태: GET + keyword).

    내용은 POST /embeddings와 같고 모양만 다르다 — 백엔드가 GET·단건·`vector` 단수 키를
    쓰기로 해서 그 계약에 맞춘 것이다. 여러 건을 한 번에 보낼 일이 생기면 POST 쪽을 쓴다.
    """
    if not manager.can_embed():
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    z = manager.encode_queries([keyword], normalize=normalize)
    return VectorSearchResponse(
        keyword=keyword,
        dim=z.shape[1],
        normalized=normalize,
        model_version=manager.model_version(),
        vector=z[0].tolist(),
    )


@router.post("/search/vector", response_model=TextVectorResponse)
async def search_vector_post(
    request: TextVectorRequest,
    manager=Depends(get_model_manager),
) -> TextVectorResponse:
    """{"text": "하늘"} → 벡터 한 줄. 같은 경로의 GET과 내용은 같고 요청 방식만 다르다.

    백엔드가 검색어를 본문으로 보내기로 해서 추가했다. GET은 URL에 검색어가 그대로
    남아 로그·프록시에 노출되는데, POST 본문은 그렇지 않다는 차이도 있다.
    """
    if not manager.can_embed():
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    text = request.value
    if not text:
        raise HTTPException(status_code=422, detail="text가 비어 있습니다.")

    z = manager.encode_queries([text], normalize=request.normalize)
    return TextVectorResponse(
        text=text,
        dim=z.shape[1],
        normalized=request.normalize,
        model_version=manager.model_version(),
        vector=z[0].tolist(),
    )
