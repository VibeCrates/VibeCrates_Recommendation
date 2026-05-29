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
