"""
MODULE 1: Project Architecture & Environment
============================================
Production FastAPI skeleton with:
- Versioned API routers
- PydanticSettings for config
- Lifespan context manager (startup/shutdown)
- Proper folder structure
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────────────
    print(f"Starting {settings.APP_NAME} in {settings.ENVIRONMENT} mode")
    # Place DB pool init, Redis connect, etc. here
    yield
    # ── SHUTDOWN ─────────────────────────────────────
    print("Shutting down — closing connections")
    # Close DB pool, Redis, etc. here


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.API_VERSION,
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.API_PREFIX)
    return app


app = create_application()


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": settings.API_VERSION, "env": settings.ENVIRONMENT}
