"""
================================================================================
MODULE 01 — FastAPI Architecture Deep Dive
================================================================================
Topics:
  L1. ASGI, Starlette & FastAPI internals
  L2. Lifespan events & application state
  L3. APIRouter — modular architecture
  L4. Layered architecture: Router → Service → Repository

Run:  uvicorn main:app --reload
Docs: http://localhost:8000/docs
================================================================================
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# LESSON 1 — ASGI Internals
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI is built on Starlette which implements the ASGI interface.
# Every incoming request becomes a (scope, receive, send) trio.
# FastAPI converts that into a Request object and routes it to your handler.
#
# The call stack for every request is:
#   Uvicorn (ASGI server)
#     → Starlette (routing, middleware)
#       → FastAPI (dependency injection, validation)
#         → Your route handler
#
# You rarely touch ASGI directly, but knowing it exists explains:
# - Why middleware can modify requests/responses at a low level
# - Why lifespan runs outside the request cycle
# - Why FastAPI can handle both HTTP and WebSockets on one server


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 2 — Lifespan Events & Application State
# ─────────────────────────────────────────────────────────────────────────────
# Use lifespan to manage resources that should be created once and reused.
# Examples: DB connection pools, HTTP clients, ML models, caches.
#
# WRONG (old pattern, deprecated):
#   @app.on_event("startup") async def startup(): ...
#
# CORRECT (lifespan context manager):
#   @asynccontextmanager
#   async def lifespan(app: FastAPI): ...

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Everything BEFORE yield runs at startup.
    Everything AFTER yield runs at shutdown.
    This is the single correct place to manage application-level resources.
    """
    print("🚀 Starting up — creating shared resources...")

    # Create a shared async HTTP client (connection pool is reused across requests)
    # Never create httpx.AsyncClient() inside a route — that opens/closes a new
    # TCP connection pool on every request, which is extremely wasteful.
    http_client = httpx.AsyncClient(timeout=30.0)

    # Attach resources to app.state so any route can access them via request.app.state
    app.state.http_client = http_client
    app.state.start_time = time.time()
    app.state.request_count = 0  # simple in-memory counter (use Redis in production)

    print("✅ Startup complete — app is ready to serve requests")

    yield  # ← application runs here, handling requests

    # Shutdown: clean up all resources gracefully
    print("🛑 Shutting down — releasing resources...")
    await http_client.aclose()
    print("✅ Shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

class Product(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(min_length=1, max_length=200)
    price: float = Field(gt=0)
    category: str


class ProductCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    price: float = Field(gt=0)
    category: str


class ProductUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    price: float | None = Field(default=None, gt=0)
    category: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 4 — Repository Layer
# ─────────────────────────────────────────────────────────────────────────────
# Repository encapsulates ALL data access logic.
# Routes and services never know how data is stored (in-memory, SQL, Redis).
# In Module 03 we replace this with a real async SQLAlchemy repository.

class ProductRepository:
    """In-memory repository — same interface as a real DB repository."""

    def __init__(self) -> None:
        self._store: dict[str, Product] = {}

    async def get_all(self) -> list[Product]:
        return list(self._store.values())

    async def get_by_id(self, product_id: str) -> Product | None:
        return self._store.get(product_id)

    async def create(self, data: ProductCreate) -> Product:
        product = Product(**data.model_dump())
        self._store[product.id] = product
        return product

    async def update(self, product_id: str, data: ProductUpdate) -> Product | None:
        product = self._store.get(product_id)
        if not product:
            return None
        # model_dump(exclude_unset=True) only returns fields the client sent
        updated = product.model_copy(update=data.model_dump(exclude_unset=True))
        self._store[product_id] = updated
        return updated

    async def delete(self, product_id: str) -> bool:
        if product_id not in self._store:
            return False
        del self._store[product_id]
        return True


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 4 — Service Layer
# ─────────────────────────────────────────────────────────────────────────────
# Service contains ALL business logic.
# It calls the repository for data, applies rules, and returns domain objects.
# Services never know about HTTP — no Request, no HTTPException here ideally.
# (We raise HTTPException here for simplicity; in strict DDD you'd use custom exceptions.)

class ProductService:
    def __init__(self, repo: ProductRepository) -> None:
        self.repo = repo

    async def list_products(self) -> list[Product]:
        return await self.repo.get_all()

    async def get_product(self, product_id: str) -> Product:
        product = await self.repo.get_by_id(product_id)
        if not product:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
        return product

    async def create_product(self, data: ProductCreate) -> Product:
        # Business rule: price must make commercial sense
        if data.price < 0.01:
            raise HTTPException(status_code=422, detail="Price too low to be valid")
        return await self.repo.create(data)

    async def update_product(self, product_id: str, data: ProductUpdate) -> Product:
        product = await self.repo.update(product_id, data)
        if not product:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
        return product

    async def delete_product(self, product_id: str) -> dict[str, str]:
        deleted = await self.repo.delete(product_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
        return {"message": f"Product {product_id} deleted"}


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 3 — APIRouter & Dependency Injection
# ─────────────────────────────────────────────────────────────────────────────
# APIRouter groups related routes. In a real app each feature is a separate file.
# Dependencies (Depends) inject objects into route handlers — the DI system
# resolves these automatically, supports async, caches per-request by default.

# Singleton repository (in production this would be scoped to DB session)
_product_repo = ProductRepository()


def get_product_repo() -> ProductRepository:
    """Dependency: provides the product repository."""
    return _product_repo


def get_product_service(
    repo: ProductRepository = Depends(get_product_repo),
) -> ProductService:
    """Dependency: provides the product service, injecting the repo."""
    return ProductService(repo)


# Router — in a real app this lives in app/features/products/router.py
products_router = APIRouter(
    prefix="/products",
    tags=["Products"],
)


@products_router.get("/", response_model=list[Product])
async def list_products(
    service: ProductService = Depends(get_product_service),
) -> list[Product]:
    """List all products. Service handles the data access."""
    return await service.list_products()


@products_router.get("/{product_id}", response_model=Product)
async def get_product(
    product_id: str,
    service: ProductService = Depends(get_product_service),
) -> Product:
    return await service.get_product(product_id)


@products_router.post("/", response_model=Product, status_code=status.HTTP_201_CREATED)
async def create_product(
    data: ProductCreate,
    service: ProductService = Depends(get_product_service),
) -> Product:
    return await service.create_product(data)


@products_router.patch("/{product_id}", response_model=Product)
async def update_product(
    product_id: str,
    data: ProductUpdate,
    service: ProductService = Depends(get_product_service),
) -> Product:
    return await service.update_product(product_id, data)


@products_router.delete("/{product_id}")
async def delete_product(
    product_id: str,
    service: ProductService = Depends(get_product_service),
) -> dict[str, str]:
    return await service.delete_product(product_id)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM ROUTER — health, metrics, app info
# ─────────────────────────────────────────────────────────────────────────────

system_router = APIRouter(tags=["System"])


@system_router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """
    Health check endpoint.
    Demonstrates accessing app.state (set in lifespan) from a route.
    """
    uptime = time.time() - request.app.state.start_time
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "request_count": request.app.state.request_count,
    }


@system_router.get("/info")
async def info(request: Request) -> dict[str, Any]:
    """Show how to use the shared HTTP client from app.state."""
    client: httpx.AsyncClient = request.app.state.http_client
    # Example: use the shared client to call an external API
    # response = await client.get("https://api.example.com/data")
    return {
        "http_client_active": not client.is_closed,
        "note": "Shared httpx.AsyncClient from lifespan — one pool for all requests",
    }


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
# This is where you wire everything together.
# In a large app: app/main.py imports routers from feature modules.

app = FastAPI(
    title="Module 01 — FastAPI Architecture",
    description="Layered architecture with lifespan, APIRouter, and DI",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# MIDDLEWARE — runs for every request
# ─────────────────────────────────────────────────────────────────────────────
# Middleware wraps the entire request/response cycle.
# This one adds a unique request ID and tracks total request count.

@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next: Any) -> Any:
    """
    Middleware runs BEFORE and AFTER every route handler.
    call_next(request) hands off to the next layer (your route).
    """
    request_id = str(uuid4())
    request.state.request_id = request_id

    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000

    # Increment request counter on app.state
    request.app.state.request_count += 1

    # Add useful headers to every response
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

    return response


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTION HANDLERS — global error formatting
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Override the default HTTPException format to return consistent JSON.
    Every error in your API has the same shape — critical for clients/agents.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions — never expose raw tracebacks to clients."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )


# Register routers
app.include_router(system_router)
app.include_router(products_router)


# ─────────────────────────────────────────────────────────────────────────────
# PRACTICE EXERCISES
# ─────────────────────────────────────────────────────────────────────────────
# 1. Add a CategoryRepository and CategoryService with its own router.
# 2. Add a /products/search?q= endpoint in the service layer.
# 3. Add a middleware that logs request method + path + status code.
# 4. Make the request_count thread-safe using asyncio.Lock.
# 5. Add a /products/category/{category} filter endpoint.
