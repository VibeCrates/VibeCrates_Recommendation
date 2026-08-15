"""
Request and Response schemas for API
"""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    query: str = Field(..., description="자연어 검색 쿼리")
    domain: Optional[Literal["movie", "music", "book"]] = Field(
        default=None, description="도메인 필터 (미지정 시 전체 도메인 통합 검색)"
    )
    top_k: int = Field(default=10, ge=1, le=100, description="반환할 추천 수")


class RecommendationItem(BaseModel):
    item_id: str
    domain: str = Field(description="movie | music | book")
    score: float = Field(description="쿼리와의 코사인 유사도")
    title: str
    extra: Optional[dict] = Field(default=None, description="도메인별 추가 정보")


class RecommendationResponse(BaseModel):
    query: str
    domain: str
    results: List[RecommendationItem]


class ItemInfoResponse(BaseModel):
    item_id: str
    domain: str
    info: dict


class HealthCheckResponse(BaseModel):
    status: str = Field(default="healthy")
    model_loaded: bool = Field(default=False)
    index_built: dict = Field(default_factory=dict, description="도메인별 인덱스 구축 여부")


class PingResponse(BaseModel):
    """백엔드 ↔ 추천 서버 통신 확인용. 모델·인덱스와 무관하게 응답한다."""
    value: int = Field(description="서버가 돌려주는 숫자. n을 보내면 그 값, 없으면 기본값")
    received: Optional[int] = Field(default=None, description="요청에 실려 온 n (없으면 null)")
    server_time: str = Field(description="서버 시각 ISO8601 — 시계·타임존 확인용")
    model_loaded: bool = Field(description="모델 적재 여부. 통신 확인 단계에서는 false여도 정상")
