"""FastAPI application factory.

This is the entry point for the API. It assembles all routes, middleware,
and lifecycle hooks into a single FastAPI application.

Run with:
    uvicorn api_gateway.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_gateway.config import APIConfig
from api_gateway.middleware.rate_limiter import RateLimiterMiddleware
from api_gateway.middleware.request_logger import RequestLoggerMiddleware
from api_gateway.routes import data_upload, forecast, health, replenishment

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle — startup and shutdown hooks."""
    logger.info("RetailLab API starting up...")
    # TODO: Load model, initialize quant_engine service, connect Redis
    yield
    logger.info("RetailLab API shutting down...")
    # TODO: Flush telemetry logs, close connections


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = APIConfig()

    app = FastAPI(
        title="RetailLab - Smart Replenishment API",
        description="Intelligent inventory replenishment system for chain supermarkets",
        version=config.api_version,
        lifespan=lifespan,
    )

    # Middleware (order matters — last added = first executed)
    app.add_middleware(RequestLoggerMiddleware)
    app.add_middleware(
        RateLimiterMiddleware,
        requests_per_minute=config.rate_limit_per_minute,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    prefix = "/api/v1"
    app.include_router(health.router, prefix=prefix)
    app.include_router(replenishment.router, prefix=prefix)
    app.include_router(data_upload.router, prefix=prefix)
    app.include_router(forecast.router, prefix=prefix)

    return app


app = create_app()
