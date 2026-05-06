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
logging.basicConfig(level=logging.INFO)
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
