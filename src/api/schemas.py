"""
Request and Response schemas for API
"""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


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
    domain: Optional[str] = Field(
        default=None,
        description="요청에 실려 온 도메인 필터. 통합 검색(도메인 미지정)이면 null",
    )
    results: List[RecommendationItem]


class ItemInfoResponse(BaseModel):
    item_id: str
    domain: str
    info: dict


class HealthCheckResponse(BaseModel):
    status: str = Field(default="healthy")
    model_loaded: bool = Field(default=False)
    index_built: dict = Field(default_factory=dict, description="도메인별 인덱스 구축 여부")
    errors: dict = Field(
        default_factory=dict,
        description="기동 시 실패한 것과 그 이유. 비어 있으면 정상. "
                    "앱은 떠 있는데 추천만 503일 때 여기를 본다",
    )


class PingResponse(BaseModel):
    """백엔드 ↔ 추천 서버 통신 확인용. 모델·인덱스와 무관하게 응답한다."""
    value: int = Field(description="서버가 돌려주는 숫자. n을 보내면 그 값, 없으면 기본값")
    received: Optional[int] = Field(default=None, description="요청에 실려 온 n (없으면 null)")
    server_time: str = Field(description="서버 시각 ISO8601 — 시계·타임존 확인용")
    model_loaded: bool = Field(description="모델 적재 여부. 통신 확인 단계에서는 false여도 정상")


class EmbeddingRequest(BaseModel):
    """자연어 쿼리 → 벡터. 백엔드가 자기 쪽 벡터 DB에서 직접 검색하는 구조용.

    `text`와 `query` 둘 다 받는다 — /search/vector는 text, 여기는 query만 받던 탓에
    백엔드가 두 경로 사이에서 422를 반복해서 맞았다. 어느 이름으로 와도 받는 편이
    왕복을 줄인다. 둘 다 문자열 하나 또는 배열을 허용한다.
    """
    query: Optional[str | List[str]] = Field(default=None, description="쿼리 문자열 하나, 또는 문자열 배열")
    text: Optional[str | List[str]] = Field(default=None, description="`query`의 별칭")
    normalize: bool = Field(
        default=True,
        description="L2 정규화 여부. True면 내적이 그대로 코사인 유사도가 된다",
    )

    @model_validator(mode="after")
    def _require_one(self):
        if self.query is None and self.text is None:
            raise ValueError("query(또는 text)가 필요합니다")
        return self

    @property
    def values(self) -> List[str]:
        raw = self.query if self.query is not None else self.text
        return [raw] if isinstance(raw, str) else list(raw)


class EmbeddingResponse(BaseModel):
    dim: int = Field(description="벡터 차원 (768)")
    normalized: bool = Field(description="L2 정규화된 벡터인지")
    model_version: str = Field(
        description="아이템 임베딩과 같은 체크포인트에서 나왔는지 대조용. 다르면 검색 결과가 무의미하다"
    )
    vectors: List[List[float]] = Field(description="입력과 같은 순서의 float 배열")


class VectorSearchResponse(BaseModel):
    """GET /search/vector 응답. 백엔드가 요청한 형태 — 키 이름이 `vector`(단수)다.

    배치가 필요하면 POST /embeddings 쪽을 쓴다(그쪽은 `vectors` 복수).
    """
    keyword: str = Field(description="요청에 실려 온 검색어. 인코딩 확인용으로 그대로 돌려준다")
    dim: int = Field(description="벡터 차원 (768)")
    normalized: bool = Field(description="L2 정규화된 벡터인지")
    model_version: str = Field(description="아이템 임베딩과 같은 체크포인트인지 대조용")
    vector: List[float] = Field(description="쿼리 임베딩 float 배열")


class TextVectorRequest(BaseModel):
    """POST /search/vector 요청. 백엔드 계약: {"text": "하늘"}

    `query`도 같이 받는다 — 연동 초기에 필드 이름이 몇 번 바뀌었고, 둘 중 무엇이 와도
    422로 되돌려보내는 것보다 받아주는 편이 왕복을 줄인다.
    """
    text: Optional[str] = Field(default=None, description="자연어 검색어")
    query: Optional[str] = Field(default=None, description="`text`의 별칭")
    normalize: bool = Field(default=True, description="L2 정규화 여부")

    @model_validator(mode="after")
    def _require_one(self):
        if not (self.text or self.query):
            raise ValueError("text(또는 query)가 필요합니다")
        return self

    @property
    def value(self) -> str:
        return (self.text or self.query or "").strip()


class TextVectorResponse(BaseModel):
    """POST /search/vector 응답. 벡터 한 줄."""
    text: str = Field(description="요청에 실려 온 검색어. 인코딩 확인용으로 그대로 돌려준다")
    dim: int = Field(description="벡터 차원 (768)")
    normalized: bool = Field(description="L2 정규화된 벡터인지")
    model_version: str = Field(description="FAKE-no-model이면 가짜, trained_model.pt@…이면 진짜")
    vector: List[float] = Field(description="쿼리 임베딩 float 배열")
