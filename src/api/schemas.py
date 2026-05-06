"""
Request and Response schemas for API
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """
    TODO: Define request schema for recommendation endpoint.
    
    Example:
        {
            "user_id": 123,
            "num_recommendations": 10,
            "filters": {}
        }
    """
    user_id: int = Field(..., description="User ID")
    num_recommendations: int = Field(default=10, description="Number of recommendations to return")
    filters: Optional[dict] = Field(default=None, description="Optional filters for recommendations")


class RecommendationResponse(BaseModel):
    """
    TODO: Define response schema for recommendation endpoint.
    
    Example:
        {
            "user_id": 123,
            "recommendations": [
                {"item_id": 1, "score": 0.95},
                {"item_id": 2, "score": 0.92}
            ]
        }
    """
    user_id: int
    recommendations: List[dict]
    message: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint."""
    status: str = Field(default="healthy")
    message: str = Field(default="API is running")
    model_loaded: bool = Field(default=False)


class UserProfileRequest(BaseModel):
    """TODO: Request schema for user profile endpoint."""
    user_id: int


class UserProfileResponse(BaseModel):
    """TODO: Response schema for user profile endpoint."""
    user_id: int
    profile: dict


class ItemInfoRequest(BaseModel):
    """TODO: Request schema for item information endpoint."""
    item_id: int


class ItemInfoResponse(BaseModel):
    """TODO: Response schema for item information endpoint."""
    item_id: int
    info: dict
