# FastAPI Production & Agentic Systems Mastery
## Complete 8-Module Course

---

## Course structure

```
fastapi_course/
├── module_01_architecture/        # Project skeleton, config, lifespan
│   ├── main.py                    # App factory + lifespan
│   ├── config.py                  # PydanticSettings
│   ├── api/v1/router.py           # Versioned routers
│   └── README.md                  # Folder structure guide
│
├── module_02_pydantic/            # Pydantic v2 data contracts
│   └── schemas.py                 # All schema patterns
│
├── module_03_async_di/            # Async + dependency injection
│   └── dependencies.py            # DB session, services, concurrent LLM
│
├── module_04_auth/                # JWT, API keys, RBAC, rate limiting
│   └── security.py
│
├── module_05_database/            # SQLAlchemy + Alembic + pgvector
│   └── models_and_repos.py
│
├── module_06_testing_observability/  # Testing, logging, tracing
│   └── observability.py
│
├── module_07_agentic_streaming/   # SSE, WebSocket, MCP server, tools
│   └── streaming_and_mcp.py
│
└── module_08_deployment/          # Docker, Nginx, CI/CD
    ├── Dockerfile
    ├── docker-compose.yml
    ├── deployment_configs.py      # Nginx, GitHub Actions, pyproject.toml
    └── README.md
```

---

## Module summaries

### Module 1 — Project Architecture & Environment
**File:** `module_01_architecture/main.py` + `config.py`

Key patterns:
- `create_application()` factory — never create the app at module level
- `@asynccontextmanager lifespan` — all startup/shutdown logic here
- `PydanticSettings` — all env vars in one place, never `os.environ`
- Versioned routers: `app.include_router(api_router, prefix="/api/v1")`

```python
# main.py — how to run
uvicorn main:app --reload --port 8000
```

---

### Module 2 — Pydantic v2 Data Contracts
**File:** `module_02_pydantic/schemas.py`

Key patterns:
- Separate `UserCreate` / `UserResponse` — never expose ORM models
- `@field_validator` for field-level validation
- `@model_validator(mode="after")` for cross-field validation
- `ConfigDict(from_attributes=True)` for ORM → Pydantic conversion
- `LLMStructuredOutput` with `strict=True` for validating LLM JSON
- `Generic[T]` for reusable `PaginatedResponse[UserResponse]`

```python
# Validate LLM output safely
result = parse_llm_output(llm_json_response)
if result is None:
    # retry with better prompt
```

---

### Module 3 — Async, Concurrency & Dependency Injection
**File:** `module_03_async_di/dependencies.py`

Key patterns:
- `async def get_db()` — yields session, auto-commits, auto-rolls back
- `Depends(get_user_service)` — service gets DB injected automatically
- `asyncio.gather(*tasks, return_exceptions=True)` — parallel LLM calls
- `asyncio.Semaphore(5)` — cap concurrent API calls
- `BackgroundTasks` — log/notify after response is already sent
- `asyncio.TaskGroup` — fail-fast when ALL results are required

```python
# Parallel LLM calls
results = await parallel_llm_calls(["analyze this", "summarize that", "classify this"])
```

---

### Module 4 — Auth, Security & Middleware
**File:** `module_04_auth/security.py`

Key patterns:
- Access token (30min) + refresh token (7 days) rotation
- `hash_api_key()` — store SHA256 hash, never raw key in DB
- `require_role("admin")` — dependency factory for RBAC
- `@limiter.limit("60/minute")` decorator on endpoints
- `RequestIDMiddleware` — every request gets a traceable UUID

```python
# Protect an endpoint
@router.delete("/users/{id}", dependencies=[Depends(require_role("admin"))])
async def delete_user(...): ...

# Typed shorthand
@router.get("/me")
async def get_me(user: AuthUser): ...  # AuthUser = Annotated[CurrentUser, Depends(get_current_user)]
```

---

### Module 5 — Database Patterns
**File:** `module_05_database/models_and_repos.py`

Key patterns:
- `TimestampMixin` + `SoftDeleteMixin` on all models
- `BaseRepository[T]` — generic CRUD, specialized repos inherit it
- `EmbeddingRepository.similarity_search()` — cosine search with pgvector
- `selectinload` for eager loading relationships
- Alembic: `alembic revision --autogenerate -m "add users"`

```python
# Vector similarity search for RAG
docs = await embedding_repo.similarity_search(
    query_embedding=embed("what is fastapi?"),
    limit=5,
    similarity_threshold=0.75
)
```

---

### Module 6 — Testing & Observability
**File:** `module_06_testing_observability/observability.py`

Key patterns:
- `register_exception_handlers(app)` — one call, all handlers registered
- Consistent error envelope: `{"success": false, "error": {...}, "request_id": "..."}`
- `pytest-asyncio` with DB session rollback per test (tests never pollute DB)
- `patch("app.services.llm_service.call_llm", new_callable=AsyncMock)` — mock LLMs
- `structlog` JSON logs in prod, pretty logs in dev
- OpenTelemetry auto-instruments FastAPI + SQLAlchemy + httpx

```bash
# Run tests
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

### Module 7 — Streaming, WebSockets & Agentic Endpoints
**File:** `module_07_agentic_streaming/streaming_and_mcp.py`

Key patterns:
- `StreamingResponse` with `text/event-stream` for SSE token streaming
- `ConnectionManager` — tracks active WebSocket connections by client ID
- `@register_tool("name")` decorator — register Python functions as agent tools
- `/mcp/tools/list` + `/mcp/tools/call` — full MCP server in 30 lines
- `/tasks/submit` → `/tasks/{id}` — async task with status polling

```python
# Register a tool agents can call
@register_tool("search_docs")
async def search_docs(query: str, limit: int = 5) -> dict:
    return await embedding_repo.similarity_search(query_embedding, limit)
```

---

### Module 8 — Deployment & DevOps
**Files:** `Dockerfile`, `docker-compose.yml`, `deployment_configs.py`

Key patterns:
- Multi-stage Dockerfile: builder → runtime (smaller final image)
- Non-root user in container (`appuser`)
- Gunicorn + UvicornWorker: `workers = (2 × CPUs) + 1`
- Nginx: rate limiting, SSL termination, SSE buffering disabled, WS upgrade
- GitHub Actions: test → lint → build → push → deploy (zero-downtime)

```bash
# Local production stack
docker compose up -d

# Run migrations after deploy
docker compose exec api alembic upgrade head

# View logs
docker compose logs api -f
```

---

## Installation — complete

```bash
pip install \
  fastapi uvicorn[standard] gunicorn \
  pydantic pydantic-settings \
  sqlalchemy asyncpg alembic pgvector \
  redis[hiredis] \
  passlib[bcrypt] PyJWT \
  httpx structlog slowapi \
  opentelemetry-sdk opentelemetry-instrumentation-fastapi \
  opentelemetry-exporter-otlp-proto-grpc \
  prometheus-client \
  pytest pytest-asyncio pytest-cov
```

---

## The production integration order

When assembling all modules into one real project:

1. `main.py` calls `create_application()` (Module 1)
2. `lifespan` initializes DB engine + Redis (Modules 3 + 5)
3. `register_exception_handlers(app)` (Module 6)
4. `app.add_middleware(RequestIDMiddleware)` (Module 6)
5. `app.state.limiter = limiter` (Module 4)
6. `configure_logging(settings.ENVIRONMENT)` (Module 6)
7. `configure_telemetry(app, ...)` (Module 6)
8. `app.include_router(auth_router)` (Module 4)
9. `app.include_router(agents_router)` (Module 7)
10. `app.include_router(mcp_router)` (Module 7)
11. Deploy via Docker Compose + GitHub Actions (Module 8)

---

## Quick reference: common patterns

```python
# Inject DB
async def endpoint(db: DBSession): ...

# Inject auth + DB
async def endpoint(user: AuthUser, db: DBSession): ...

# Admin only
@router.post("/admin/action", dependencies=[AdminRequired])
async def admin_action(): ...

# Stream LLM to frontend
return StreamingResponse(stream_groq_response(messages, model), media_type="text/event-stream")

# Call tool from agent
POST /api/v1/tools/call  {"tool_name": "search_docs", "arguments": {"query": "fastapi async"}}

# MCP tool call from Claude
POST /mcp/tools/call  {"name": "search_knowledge_base", "arguments": {"query": "..."}}
```
