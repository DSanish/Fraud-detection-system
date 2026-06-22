"""
Fraud Detection API — FastAPI Application Entry Point
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.predict import router
from api.services.prediction_service import prediction_service
from database.connection import create_tables

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── App Lifecycle ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    logger.info("🚀 Starting Fraud Detection API...")

    # Create DB tables
    try:
        create_tables()
        logger.info("✅ Database tables ready.")
    except Exception as e:
        logger.warning(f"Database setup failed: {e}")

    # Load ML model
    model_dir = os.getenv("MODEL_DIR", "models")
    prediction_service.load(model_dir=model_dir)
    if prediction_service.is_ready:
        logger.info("✅ Fraud detection model loaded.")
    else:
        logger.warning("⚠️  Model not loaded — running in rule-based mode.")

    yield

    logger.info("🛑 Shutting down Fraud Detection API.")


# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time fraud detection for banking and UPI transactions. "
        "Powered by XGBoost machine learning model with >95% ROC-AUC."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request Logging Middleware ───────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} [{elapsed:.1f}ms]")
    return response


# ─── Global Exception Handler ────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )


# ─── Include Routers ─────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


# ─── Root Endpoint ────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "predict": "/api/v1/predict",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        workers=int(os.getenv("API_WORKERS", 1)),
    )