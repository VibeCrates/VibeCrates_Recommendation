"""
FastAPI Application - Main API Server
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from .routes import router
from .dependencies import initialize_dependencies, cleanup_dependencies
from src import config

# Configure logging
# 모든 줄에 시각을 찍는다. uvicorn 기본 접근 로그에는 시각이 없어 "백엔드가 마지막으로
# 부른 때"를 파일 수정 시각으로 짐작해야 했다(10/14). uvicorn은 앱을 import하기 전에
# 자기 로거를 설정하므로, 여기서 그 핸들러의 포맷만 바꾼다.
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
for _name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    for _h in logging.getLogger(_name).handlers:
        _h.setFormatter(logging.Formatter(LOG_FORMAT))
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="VibeCrates Recommendation System",
    description="API for recommendation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Called when FastAPI app starts."""
    logger.info("Starting VibeCrates Recommendation API...")
    await initialize_dependencies()
    logger.info("Startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Called when FastAPI app shuts down."""
    logger.info("Shutting down VibeCrates Recommendation API...")
    await cleanup_dependencies()
    logger.info("Shutdown complete")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to VibeCrates Recommendation System",
        "docs": "/docs",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD
    )
