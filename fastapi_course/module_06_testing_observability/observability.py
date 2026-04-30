"""
MODULE 6: Error Handling, Testing & Observability
==================================================
- Global exception handlers
- Structured error response envelope
- pytest-asyncio tests with TestClient + httpx
- Mocking LLM calls
- structlog JSON logging
- OpenTelemetry tracing
- Prometheus metrics
"""

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Exception handlers (add to main.py)
# ─────────────────────────────────────────────────────────────────────────────

import uuid
import time
from typing import Any
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail
    request_id: str
    timestamp: float


def make_error_response(
    request: Request,
    code: str,
    message: str,
    status_code: int,
    field: str | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {"code": code, "message": message, "field": field},
            "request_id": request_id,
            "timestamp": time.time(),
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Call this in main.py after creating the app."""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        first_error = errors[0] if errors else {}
        field = ".".join(str(x) for x in first_error.get("loc", [])[1:])
        message = first_error.get("msg", "Validation error")
        return make_error_response(request, "VALIDATION_ERROR", message, 422, field or None)

    from fastapi import HTTPException

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return make_error_response(request, "HTTP_ERROR", exc.detail, exc.status_code)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        # Log the full traceback here
        import traceback
        traceback.print_exc()
        return make_error_response(
            request, "INTERNAL_ERROR",
            "An unexpected error occurred",
            status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Middleware — request ID + timing
# ─────────────────────────────────────────────────────────────────────────────

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        start = time.time()
        response = await call_next(request)
        duration_ms = round((time.time() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms}ms"
        return response


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Structured logging with structlog
# ─────────────────────────────────────────────────────────────────────────────

import structlog

def configure_logging(environment: str) -> None:
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if environment == "production":
        # JSON logs for log aggregation (Datadog, CloudWatch, etc.)
        structlog.configure(
            processors=shared_processors + [structlog.processors.JSONRenderer()],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
    else:
        # Human-readable in dev
        structlog.configure(
            processors=shared_processors + [structlog.dev.ConsoleRenderer()],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
        )


logger = structlog.get_logger()

# Usage:
# await logger.ainfo("user_created", user_id=str(user.id), email=user.email)
# await logger.aerror("llm_call_failed", error=str(e), prompt_tokens=512)


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: OpenTelemetry tracing
# ─────────────────────────────────────────────────────────────────────────────

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor


def configure_telemetry(app: FastAPI, service_name: str, otlp_endpoint: str) -> None:
    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
    )
    trace.set_tracer_provider(provider)

    # Auto-instrument FastAPI, SQLAlchemy, httpx
    FastAPIInstrumentor.instrument_app(app)
    SQLAlchemyInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()


tracer = trace.get_tracer(__name__)


async def call_llm_with_trace(prompt: str, model: str) -> str:
    """Example: manual span around LLM call."""
    with tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.prompt_length", len(prompt))

        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post("https://api.groq.com/...", json={"prompt": prompt})
            result = response.json()

        span.set_attribute("llm.completion_tokens", result.get("usage", {}).get("completion_tokens", 0))
        return result["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# PART 5: Prometheus metrics
# ─────────────────────────────────────────────────────────────────────────────

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "Request latency", ["endpoint"])
LLM_CALL_COUNT = Counter("llm_calls_total", "Total LLM API calls", ["model", "status"])
LLM_CALL_LATENCY = Histogram("llm_call_duration_seconds", "LLM call latency", ["model"])


async def metrics_endpoint(request: Request) -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# In main.py: app.add_route("/metrics", metrics_endpoint)


# ─────────────────────────────────────────────────────────────────────────────
# PART 6: pytest-asyncio tests
# ─────────────────────────────────────────────────────────────────────────────

# tests/conftest.py
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.main import app
from app.db.session import get_db
from app.models import Base

TEST_DATABASE_URL = "postgresql+asyncpg://user:pass@localhost:5432/testdb"

@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest_asyncio.fixture
async def db_session(test_engine):
    AsyncTestSession = async_sessionmaker(test_engine, expire_on_commit=False)
    async with AsyncTestSession() as session:
        yield session
        await session.rollback()  # rollback after each test

@pytest_asyncio.fixture
async def client(db_session):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
"""

# tests/test_users.py
"""
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_create_user(client: AsyncClient):
    response = await client.post("/api/v1/users/", json={
        "email": "test@example.com",
        "password": "SecurePass1",
        "full_name": "Test User",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "id" in data
    assert "password" not in data  # never leak password

@pytest.mark.asyncio
async def test_login(client: AsyncClient):
    response = await client.post("/api/v1/auth/login", json={
        "email": "test@example.com",
        "password": "SecurePass1",
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
"""

# tests/test_llm_endpoints.py — mocking LLM calls
"""
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_analyze_text_mocks_llm(client: AsyncClient):
    mock_response = {
        "choices": [{"message": {"content": "This is a positive review."}}],
        "usage": {"completion_tokens": 8}
    }

    with patch("app.services.llm_service.call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_response
        response = await client.post("/api/v1/agents/analyze", json={"text": "Great product!"})

    assert response.status_code == 200
    assert "analysis" in response.json()
    mock_llm.assert_called_once()
"""

# pytest.ini
"""
[pytest]
asyncio_mode = auto
testpaths = tests
"""
