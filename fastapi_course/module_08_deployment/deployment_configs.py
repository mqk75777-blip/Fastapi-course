# MODULE 8: Production Deployment
# =================================
# This file contains all deployment configs.
# Copy relevant sections to your project root.

# ──────────────────────────────────────────────────────────────────────────────
# FILE: Dockerfile (multi-stage, production-ready)
# ──────────────────────────────────────────────────────────────────────────────
DOCKERFILE = """
# Stage 1: build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system deps for psycopg2, cryptography, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential libpq-dev curl && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Stage 2: runtime image (minimal)
FROM python:3.12-slim AS runtime

# Non-root user for security
RUN groupadd --gid 1001 appgroup && \\
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy only what's needed from builder
COPY --from=builder /app/.venv /app/.venv
COPY --chown=appuser:appgroup . .

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8000/health || exit 1

# Gunicorn with Uvicorn workers
CMD ["gunicorn", "main:app", \\
     "--worker-class", "uvicorn.workers.UvicornWorker", \\
     "--workers", "4", \\
     "--bind", "0.0.0.0:8000", \\
     "--timeout", "120", \\
     "--keepalive", "5", \\
     "--access-logfile", "-", \\
     "--error-logfile", "-"]
"""

# Rule: workers = (2 × CPU cores) + 1  →  4-core server = 9 workers
# For async FastAPI: 2-4 workers is usually enough (async handles concurrency)


# ──────────────────────────────────────────────────────────────────────────────
# FILE: docker-compose.yml (full production stack)
# ──────────────────────────────────────────────────────────────────────────────
DOCKER_COMPOSE = """
version: '3.9'

services:
  api:
    build:
      context: .
      target: runtime
    container_name: fastapi_app
    restart: unless-stopped
    env_file: .env
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}
      - REDIS_URL=redis://redis:6379/0
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - backend

  db:
    image: pgvector/pgvector:pg16
    container_name: fastapi_postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - backend

  redis:
    image: redis:7-alpine
    container_name: fastapi_redis
    restart: unless-stopped
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - backend

  nginx:
    image: nginx:alpine
    container_name: fastapi_nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    networks:
      - backend

volumes:
  postgres_data:
  redis_data:

networks:
  backend:
    driver: bridge
"""


# ──────────────────────────────────────────────────────────────────────────────
# FILE: nginx/nginx.conf
# ──────────────────────────────────────────────────────────────────────────────
NGINX_CONF = """
worker_processes auto;

events {
    worker_connections 1024;
}

http {
    upstream fastapi {
        server api:8000;
        keepalive 32;
    }

    # Rate limiting zone
    limit_req_zone $binary_remote_addr zone=api:10m rate=60r/m;

    server {
        listen 80;
        server_name yourdomain.com;
        return 301 https://$host$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;

        # Security headers
        add_header Strict-Transport-Security "max-age=31536000" always;
        add_header X-Content-Type-Options nosniff;
        add_header X-Frame-Options DENY;

        location / {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://fastapi;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_http_version 1.1;
            proxy_set_header Connection "";  # enable keepalive
        }

        # SSE — disable buffering
        location /api/v1/chat/stream {
            proxy_pass http://fastapi;
            proxy_buffering off;
            proxy_cache off;
            proxy_set_header Connection keep-alive;
        }

        # WebSocket
        location /api/v1/ws/ {
            proxy_pass http://fastapi;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_read_timeout 86400;
        }
    }
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# FILE: .github/workflows/deploy.yml (CI/CD pipeline)
# ──────────────────────────────────────────────────────────────────────────────
GITHUB_ACTIONS = """
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync --frozen

      - name: Run linting
        run: |
          uv run ruff check .
          uv run mypy app/ --ignore-missing-imports

      - name: Run tests
        env:
          DATABASE_URL: postgresql+asyncpg://postgres:testpass@localhost:5432/testdb
          REDIS_URL: redis://localhost:6379/0
          SECRET_KEY: test-secret-key-for-ci
        run: uv run pytest tests/ -v --cov=app --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          target: runtime
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - name: Deploy to Railway
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
        run: |
          npm install -g @railway/cli
          railway up --service api

      # OR deploy to VPS via SSH:
      # - name: Deploy to VPS
      #   uses: appleboy/ssh-action@v1
      #   with:
      #     host: ${{ secrets.VPS_HOST }}
      #     username: deploy
      #     key: ${{ secrets.VPS_SSH_KEY }}
      #     script: |
      #       cd /app
      #       docker compose pull
      #       docker compose up -d --no-build
      #       docker compose exec api alembic upgrade head
"""


# ──────────────────────────────────────────────────────────────────────────────
# FILE: pyproject.toml
# ──────────────────────────────────────────────────────────────────────────────
PYPROJECT_TOML = """
[project]
name = "fastapi-production-app"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "gunicorn>=23.0.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.6.0",
    "sqlalchemy>=2.0.36",
    "asyncpg>=0.30.0",
    "alembic>=1.14.0",
    "pgvector>=0.3.6",
    "redis[hiredis]>=5.2.0",
    "passlib[bcrypt]>=1.7.4",
    "PyJWT>=2.10.0",
    "httpx>=0.27.0",
    "structlog>=24.4.0",
    "slowapi>=0.1.9",
    "opentelemetry-sdk>=1.28.0",
    "opentelemetry-instrumentation-fastapi>=0.49b0",
    "opentelemetry-instrumentation-sqlalchemy>=0.49b0",
    "opentelemetry-instrumentation-httpx>=0.49b0",
    "opentelemetry-exporter-otlp-proto-grpc>=1.28.0",
    "prometheus-client>=0.21.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "httpx>=0.27.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.12"
strict = true
"""

print("Module 8 deployment configs loaded.")
print("Files to create in your project:")
print("  Dockerfile")
print("  docker-compose.yml")
print("  nginx/nginx.conf")
print("  .github/workflows/deploy.yml")
print("  pyproject.toml")
