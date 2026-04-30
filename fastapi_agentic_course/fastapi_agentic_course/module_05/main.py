"""
================================================================================
MODULE 05 — Advanced Dependency Injection & Middleware
================================================================================
Topics:
  L1. Dependency graphs & sub-dependencies
  L2. Request-scoped context with contextvars
  L3. Custom middleware — auth, logging, tracing
  L4. dependency_overrides for testing

Run:  uvicorn main:app --reload
Docs: http://localhost:8000/docs
================================================================================
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator, Callable
from contextvars import ContextVar
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.testclient import TestClient
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 2 — ContextVar for Request-Scoped State
# ─────────────────────────────────────────────────────────────────────────────
# ContextVar stores values that are LOCAL to the current async task.
# When two requests run concurrently, each has its own request_id — they
# don't interfere with each other, even though they share the same process.
# This is the async-safe equivalent of threading.local().

request_id_var: ContextVar[str] = ContextVar("request_id", default="unknown")
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="default")


def get_request_id() -> str:
    """Read the request ID for the current request from context."""
    return request_id_var.get()


def get_current_user_id() -> str | None:
    """Read the user ID set by the auth middleware."""
    return user_id_var.get()


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 3 — Custom Middleware
# ─────────────────────────────────────────────────────────────────────────────
# Two options:
#   1. BaseHTTPMiddleware  — simple, works with Starlette/FastAPI naturally
#   2. Pure ASGI middleware — faster (no overhead), more complex to write
#
# Performance note: BaseHTTPMiddleware has a known overhead due to wrapping.
# For high-throughput APIs (10k+ req/s), consider pure ASGI middleware.

class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    LESSON 3 — Sets request-scoped context vars for every request.
    Runs BEFORE any dependency or route handler.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique ID for this request
        req_id = str(uuid.uuid4())

        # Set context vars — these are visible to all code in this request's async task
        request_id_token = request_id_var.set(req_id)
        tenant_token = tenant_id_var.set(
            request.headers.get("X-Tenant-ID", "default")
        )

        # Store on request.state too for middleware that reads headers
        request.state.request_id = req_id

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            # ALWAYS reset ContextVars — prevents state leaking between requests
            request_id_var.reset(request_id_token)
            tenant_id_var.reset(tenant_token)


class TimingMiddleware(BaseHTTPMiddleware):
    """
    LESSON 3 — Adds X-Process-Time-Ms header to all responses.
    Demonstrates chaining multiple middlewares.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        return response


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    LESSON 3 — Logs every request in a structured format.
    In production: use structlog (see Module 08) instead of print.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        # Structured log line — use structlog.info() in production
        print({
            "event": "http_request",
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "request_id": get_request_id(),
            "tenant_id": tenant_id_var.get(),
        })
        return response


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 1 — Dependency Graph
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI builds a dependency graph and resolves it in the correct order.
# Sub-dependencies are cached: if two deps require get_db(), the session
# is created ONCE per request and shared.

# Simulated config
class AppConfig(BaseModel):
    db_url: str = "sqlite+aiosqlite:///./app.db"
    redis_url: str = "redis://localhost:6379"
    debug: bool = True
    max_page_size: int = 100


_config = AppConfig()


def get_config() -> AppConfig:
    """
    Root dependency — provides app configuration.
    Cached for the app lifetime (not per-request) since it's not a generator.
    """
    return _config


# Simulated DB session
class FakeDBSession:
    def __init__(self, url: str) -> None:
        self.url = url
        self.queries: list[str] = []

    async def execute(self, sql: str) -> list[dict]:
        self.queries.append(sql)
        return [{"result": "fake_data"}]


async def get_db(config: AppConfig = Depends(get_config)) -> AsyncGenerator[FakeDBSession, None]:
    """
    LESSON 1 — Generator dependency.
    yield suspends here, handler runs, then cleanup executes.
    FastAPI calls this once per request due to generator caching.
    """
    session = FakeDBSession(url=config.db_url)
    print(f"[DB] Opening session for request {get_request_id()}")
    try:
        yield session
    finally:
        # Cleanup always runs — even if an exception occurred
        print(f"[DB] Closing session. Executed {len(session.queries)} queries.")


# Simulated cache
class FakeCache:
    def __init__(self, url: str) -> None:
        self.url = url
        self._store: dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        return self._store.get(key)

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        self._store[key] = value


async def get_cache(config: AppConfig = Depends(get_config)) -> FakeCache:
    """
    LESSON 1 — Non-generator dependency.
    No cleanup needed — the cache client is stateless between requests.
    In production: return the global aioredis client from app.state.
    """
    return FakeCache(url=config.redis_url)


# Service that depends on BOTH db AND cache
class ProductService:
    def __init__(self, db: FakeDBSession, cache: FakeCache, config: AppConfig) -> None:
        self.db = db
        self.cache = cache
        self.config = config

    async def get_product(self, product_id: str) -> dict[str, Any]:
        # Cache-aside pattern
        cached = await self.cache.get(f"product:{product_id}")
        if cached:
            return {**cached, "source": "cache"}

        results = await self.db.execute(f"SELECT * FROM products WHERE id = '{product_id}'")
        if not results:
            raise HTTPException(status_code=404, detail="Product not found")

        product = results[0]
        await self.cache.set(f"product:{product_id}", product)
        return {**product, "source": "database"}


async def get_product_service(
    db: FakeDBSession = Depends(get_db),
    cache: FakeCache = Depends(get_cache),
    config: AppConfig = Depends(get_config),
) -> ProductService:
    """
    LESSON 1 — Complex dependency with sub-dependencies.
    FastAPI resolves get_db, get_cache, get_config, then constructs this.
    The dependency graph looks like:
      get_product_service
        ├── get_db
        │     └── get_config  ← shared instance
        ├── get_cache
        │     └── get_config  ← SAME shared instance (cached)
        └── get_config        ← SAME shared instance (cached)
    """
    return ProductService(db=db, cache=cache, config=config)


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 2 — ContextVar-based dependencies
# ─────────────────────────────────────────────────────────────────────────────

def get_request_context() -> dict[str, str | None]:
    """
    Dependencies can read ContextVars set by middleware.
    No need to pass request_id through every function call.
    """
    return {
        "request_id": request_id_var.get(),
        "tenant_id": tenant_id_var.get(),
        "user_id": user_id_var.get(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# APP ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Module 05 — Advanced DI & Middleware",
    description="Dependency graphs, ContextVars, custom middleware",
    version="1.0.0",
)

# Middleware order matters: LAST added = OUTERMOST (runs first)
# Request flow: StructuredLogging → Timing → RequestContext → Route
app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestContextMiddleware)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/products/{product_id}", tags=["Products"])
async def get_product(
    product_id: str,
    service: ProductService = Depends(get_product_service),
    ctx: dict = Depends(get_request_context),
) -> dict[str, Any]:
    """
    Demonstrates the full dependency graph in action.
    Check the terminal: see DB session open/close, query count.
    Check the response headers: X-Request-ID, X-Process-Time-Ms.
    """
    product = await service.get_product(product_id)
    return {
        "product": product,
        "request_context": ctx,
    }


@app.get("/context", tags=["Debug"])
async def show_context(ctx: dict = Depends(get_request_context)) -> dict:
    """Shows what's in the request context. Add X-Tenant-ID header to test."""
    return ctx


@app.get("/config", tags=["Debug"])
async def show_config(config: AppConfig = Depends(get_config)) -> AppConfig:
    """Shows the app config. Notice get_config is called once — same instance."""
    return config


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 4 — dependency_overrides for Testing
# ─────────────────────────────────────────────────────────────────────────────
# This is the most powerful testing pattern in FastAPI.
# You override ANY dependency with a fake — no mocking libraries needed.

def test_dependency_overrides():
    """
    Demonstrates dependency_overrides.
    Run this function directly: python main.py
    """

    # Fake DB that returns predictable data
    class FakeTestDB(FakeDBSession):
        def __init__(self) -> None:
            super().__init__(url="sqlite:///:memory:")

        async def execute(self, sql: str) -> list[dict]:
            return [{"result": "test_product", "id": "test-123"}]

    async def override_get_db():
        """Replace the real DB with a fake one for all tests."""
        yield FakeTestDB()

    def override_get_config():
        """Inject a test config."""
        return AppConfig(debug=True, db_url="sqlite:///:memory:")

    # Apply overrides BEFORE creating the test client
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_config] = override_get_config

    client = TestClient(app)

    response = client.get("/products/any-id-works-now")
    assert response.status_code == 200
    data = response.json()
    assert data["product"]["result"] == "test_product"
    print("✅ dependency_overrides test passed!")

    # IMPORTANT: clean up overrides after the test
    app.dependency_overrides.clear()


# ─────────────────────────────────────────────────────────────────────────────
# PURE ASGI MIDDLEWARE EXAMPLE (faster than BaseHTTPMiddleware)
# ─────────────────────────────────────────────────────────────────────────────
# Uncomment and use add_middleware below for high-performance scenarios

class PureASGIMiddleware:
    """
    Pure ASGI middleware — no Starlette overhead.
    About 20-30% faster than BaseHTTPMiddleware for high-throughput APIs.
    Trade-off: more boilerplate, harder to read.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers[b"x-powered-by"] = b"FastAPI-Agentic"
                message["headers"] = list(headers.items())
            await send(message)

        await self.app(scope, receive, send_wrapper)

# app.add_middleware(PureASGIMiddleware)  # uncomment to use


if __name__ == "__main__":
    test_dependency_overrides()

# ─────────────────────────────────────────────────────────────────────────────
# PRACTICE EXERCISES
# ─────────────────────────────────────────────────────────────────────────────
# 1. Add a RateLimitMiddleware that limits to 10 req/s per IP using ContextVar.
# 2. Create a dependency that reads user_id from a JWT and sets user_id_var.
# 3. Write a test using dependency_overrides to inject a fake ProductService.
# 4. Add a timeout middleware that returns 504 if a request takes > 5s.
# 5. Implement a circuit breaker dependency for calling external AI APIs.
