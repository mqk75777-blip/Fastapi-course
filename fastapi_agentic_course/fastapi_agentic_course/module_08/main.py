"""
================================================================================
MODULE 08 — Observability, Security & Production Deployment
================================================================================
Topics:
  L1. Structured logging with structlog
  L2. OpenTelemetry tracing
  L3. Security hardening — headers, CORS, input validation
  L4. Health checks — liveness & readiness
  L5. Docker + Kubernetes deployment config

Run:  uvicorn main:app --reload
Docs: http://localhost:8000/docs
================================================================================
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 1 — Structured Logging with structlog
# ─────────────────────────────────────────────────────────────────────────────
# Structured logging = every log is a machine-readable JSON event.
# Tools like Datadog, Grafana Loki, and CloudWatch can query these fields.
# Never use print() or raw logging.basicConfig in production.

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,       # merge request_id, user_id from context
        structlog.stdlib.add_log_level,                # add "level" field
        structlog.stdlib.add_logger_name,              # add "logger" field
        structlog.processors.TimeStamper(fmt="iso"),   # ISO 8601 timestamps
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()                # pretty in dev; use JSONRenderer in prod
        # Production: structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 2 — OpenTelemetry Tracing (manual instrumentation)
# ─────────────────────────────────────────────────────────────────────────────
# OpenTelemetry is the industry standard for distributed tracing.
# In production: pip install opentelemetry-instrumentation-fastapi
# and add FastAPIInstrumentor().instrument_app(app) to lifespan.
# Here we show manual span creation for custom business logic tracing.

# Simulate OTel span (real implementation shown in comments)
class FakeSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value
        logger.debug("span.attribute", span=self.name, key=key, value=str(value))

    def __enter__(self) -> FakeSpan:
        logger.debug("span.start", name=self.name)
        return self

    def __exit__(self, *args: Any) -> None:
        logger.debug("span.end", name=self.name, attributes=self.attributes)


def create_span(name: str) -> FakeSpan:
    """
    Production version:
        from opentelemetry import trace
        tracer = trace.get_tracer(__name__)
        return tracer.start_as_current_span(name)
    """
    return FakeSpan(name)


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK STORE — simulates dependency states
# ─────────────────────────────────────────────────────────────────────────────

class DependencyHealth:
    def __init__(self) -> None:
        self._states: dict[str, bool] = {
            "database": True,
            "redis": True,
            "groq_api": True,
        }

    def mark_unhealthy(self, dep: str) -> None:
        self._states[dep] = False

    def mark_healthy(self, dep: str) -> None:
        self._states[dep] = True

    @property
    def all_healthy(self) -> bool:
        return all(self._states.values())

    @property
    def states(self) -> dict[str, bool]:
        return dict(self._states)


health_tracker = DependencyHealth()


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("app.startup", version="1.0.0", environment="production")

    # Production OTel setup (uncomment when using real OTel):
    # from opentelemetry.sdk.trace import TracerProvider
    # from opentelemetry.sdk.trace.export import BatchSpanProcessor
    # from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    # from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    # provider = TracerProvider()
    # provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    # trace.set_tracer_provider(provider)
    # FastAPIInstrumentor().instrument_app(app)

    app.state.start_time = time.time()
    logger.info("app.ready")
    yield

    logger.info("app.shutdown")


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 1 — Logging Middleware
# ─────────────────────────────────────────────────────────────────────────────

class LoggingMiddleware(BaseHTTPMiddleware):
    """Binds request_id to structlog context for every log in this request."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Bind fields to structlog context — visible in ALL logs during this request
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
        )

        start = time.perf_counter()
        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "http.request",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception("http.request.error", duration_ms=round(duration_ms, 2), error=str(exc))
            raise


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 3 — Security Headers Middleware
# ─────────────────────────────────────────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds OWASP-recommended security headers to every response.
    These prevent common web attacks even if your code has vulnerabilities.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Prevents clickjacking attacks
        response.headers["X-Frame-Options"] = "DENY"

        # Prevents MIME type sniffing (stops XSS via file upload)
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Forces HTTPS for 1 year (only enable in production with real HTTPS)
        # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Controls what browser features are available
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # Content Security Policy — restricts where resources can be loaded from
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "object-src 'none';"
        )

        # Referrer policy — controls what's sent in Referer header
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Remove server fingerprinting
        response.headers.pop("server", None)

        return response


# ─────────────────────────────────────────────────────────────────────────────
# APP ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Module 08 — Observability & Security",
    description="Structured logging, OTel tracing, security headers, health checks",
    version="1.0.0",
    lifespan=lifespan,
    # Disable /docs and /redoc in production
    # docs_url=None,
    # redoc_url=None,
)

# LESSON 3 — CORS: explicitly list allowed origins
# Never use allow_origins=["*"] in production — it allows any website to
# make requests to your API using the visitor's credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # local dev frontend
        "https://yourdomain.com",   # production frontend
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Tenant-ID"],
    expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    max_age=600,  # preflight cache duration in seconds
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(LoggingMiddleware)


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 4 — Health Check Endpoints
# ─────────────────────────────────────────────────────────────────────────────
# Kubernetes uses TWO health probes:
#   Liveness  (/health/live)  — is the process alive? If fails: restart container.
#   Readiness (/health/ready) — can it serve traffic? If fails: stop sending requests.
#
# A pod can be alive but not ready (e.g., warming up, DB not connected yet).

async def check_database() -> tuple[bool, float]:
    """Simulate checking DB connectivity."""
    start = time.perf_counter()
    await asyncio.sleep(0.001)  # simulate query
    latency_ms = (time.perf_counter() - start) * 1000
    return health_tracker.states["database"], latency_ms


async def check_redis() -> tuple[bool, float]:
    """Simulate checking Redis connectivity."""
    start = time.perf_counter()
    await asyncio.sleep(0.001)
    latency_ms = (time.perf_counter() - start) * 1000
    return health_tracker.states["redis"], latency_ms


async def check_groq_api() -> tuple[bool, float]:
    """Simulate checking external API connectivity."""
    start = time.perf_counter()
    await asyncio.sleep(0.002)
    latency_ms = (time.perf_counter() - start) * 1000
    return health_tracker.states["groq_api"], latency_ms


@app.get("/health/live", tags=["Health"])
async def liveness() -> dict[str, str]:
    """
    LIVENESS probe — Is the process running?
    Keep this FAST and SIMPLE — only check if the process can respond.
    Never check external dependencies here (DB could be down for 10s — don't kill the pod).
    """
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
async def readiness(request: Request) -> dict[str, Any]:
    """
    READINESS probe — Can the app serve real traffic?
    Check ALL dependencies needed to serve requests.
    Returns 503 if any dependency is unhealthy.

    Kubernetes readiness probe config:
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          failureThreshold: 3
    """
    # Run all checks concurrently
    db_ok, db_latency = await check_database()
    redis_ok, redis_latency = await check_redis()
    groq_ok, groq_latency = await check_groq_api()

    uptime = time.time() - request.app.state.start_time

    checks = {
        "database": {"healthy": db_ok, "latency_ms": round(db_latency, 2)},
        "redis": {"healthy": redis_ok, "latency_ms": round(redis_latency, 2)},
        "groq_api": {"healthy": groq_ok, "latency_ms": round(groq_latency, 2)},
    }

    all_healthy = all(c["healthy"] for c in checks.values())

    response_body = {
        "status": "ready" if all_healthy else "degraded",
        "uptime_seconds": round(uptime, 2),
        "checks": checks,
    }

    if not all_healthy:
        logger.warning("health.check.failed", checks=checks)
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content=response_body)

    return response_body


@app.get("/health/metrics", tags=["Health"])
async def metrics(request: Request) -> dict[str, Any]:
    """
    Prometheus-style metrics endpoint (use prometheus_client in production).
    pip install prometheus-fastapi-instrumentator
    """
    return {
        "uptime_seconds": round(time.time() - request.app.state.start_time, 2),
        "version": "1.0.0",
    }


# ─────────────────────────────────────────────────────────────────────────────
# DEMO ROUTES
# ─────────────────────────────────────────────────────────────────────────────

class ItemCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)


@app.post("/items", status_code=201, tags=["Items"])
async def create_item(data: ItemCreate) -> dict[str, Any]:
    """Demonstrates structured logging with business context."""
    item_id = str(uuid.uuid4())

    with create_span("item.create") as span:
        span.set_attribute("item.id", item_id)
        span.set_attribute("item.name", data.name)

        # This log automatically includes request_id from structlog context
        logger.info("item.created", item_id=item_id, name=data.name)

    return {"id": item_id, "name": data.name, "description": data.description}


@app.post("/health/simulate-failure/{dependency}", tags=["Health"])
async def simulate_failure(dependency: str) -> dict[str, str]:
    """For testing — simulates a dependency going down."""
    if dependency not in ["database", "redis", "groq_api"]:
        raise HTTPException(status_code=400, detail=f"Unknown dependency: {dependency}")
    health_tracker.mark_unhealthy(dependency)
    logger.warning("health.dependency.failed", dependency=dependency)
    return {"message": f"Simulated failure: {dependency}"}


@app.post("/health/restore/{dependency}", tags=["Health"])
async def restore_dependency(dependency: str) -> dict[str, str]:
    health_tracker.mark_healthy(dependency)
    return {"message": f"Restored: {dependency}"}


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 5 — Dockerfile (save as Dockerfile in your project root)
# ─────────────────────────────────────────────────────────────────────────────
DOCKERFILE = """
# Multi-stage build — final image has NO build tools (smaller, more secure)
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS runtime
WORKDIR /app

# Run as non-root user — security best practice
RUN addgroup --system app && adduser --system --group app
USER app

COPY --from=builder /root/.local /home/app/.local
COPY . .

ENV PATH=/home/app/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Use multiple workers in production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
"""

# ─────────────────────────────────────────────────────────────────────────────
# LESSON 5 — Kubernetes Deployment (save as k8s-deployment.yaml)
# ─────────────────────────────────────────────────────────────────────────────
K8S_DEPLOYMENT = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-app
  labels:
    app: fastapi-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fastapi-app
  template:
    metadata:
      labels:
        app: fastapi-app
    spec:
      containers:
        - name: fastapi
          image: your-registry/fastapi-app:latest
          ports:
            - containerPort: 8000
          env:
            - name: SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: secret-key
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: database-url
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
            failureThreshold: 3
---
apiVersion: v1
kind: Service
metadata:
  name: fastapi-service
spec:
  selector:
    app: fastapi-app
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fastapi-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fastapi-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
"""


@app.get("/deployment/configs", tags=["DevOps"])
async def get_deployment_configs() -> dict[str, str]:
    """Returns Dockerfile and K8s config as strings — study these!"""
    return {
        "dockerfile": DOCKERFILE.strip(),
        "kubernetes": K8S_DEPLOYMENT.strip(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRACTICE EXERCISES
# ─────────────────────────────────────────────────────────────────────────────
# 1. Add prometheus_fastapi_instrumentator and expose /metrics endpoint.
# 2. Add a middleware that rejects requests with bodies larger than 1MB.
# 3. Implement request ID propagation to outgoing httpx requests (X-Request-ID header).
# 4. Add structured logging for all HTTPExceptions with their status codes.
# 5. Write a Dockerfile that copies only necessary files (use .dockerignore).
