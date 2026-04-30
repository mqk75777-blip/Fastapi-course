"""
MODULE 3: Async, Concurrency & Dependency Injection
====================================================
- async/await best practices in FastAPI
- Dependency injection for DB, auth, services
- Concurrent LLM calls with asyncio.gather / TaskGroup
- Background tasks
- Connection pooling
"""

import asyncio
import time
from typing import Annotated, AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings

router = APIRouter()

# ─── Database session factory ─────────────────────────────────────────────────

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,          # detect dead connections
    pool_recycle=3600,           # recycle connections every hour
    echo=settings.DEBUG,         # log SQL in dev only
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,      # avoid lazy-load errors after commit
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a DB session per request.
    Session is automatically closed when request ends.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Type alias for cleaner endpoint signatures
DBSession = Annotated[AsyncSession, Depends(get_db)]


# ─── Service-level dependency injection ───────────────────────────────────────

class UserService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user(self, user_id: UUID):
        # DB query here (Module 5 adds real queries)
        return {"id": str(user_id), "email": "user@example.com"}

    async def create_user(self, data: dict):
        # Business logic + DB write
        return {"id": "new-uuid", **data}


def get_user_service(db: DBSession) -> UserService:
    return UserService(db)


UserServiceDep = Annotated[UserService, Depends(get_user_service)]


# ─── Request-scoped state via middleware ──────────────────────────────────────

async def get_request_id(request: Request) -> str:
    return request.state.request_id  # set by RequestIDMiddleware (Module 6)


RequestID = Annotated[str, Depends(get_request_id)]


# ─── Concurrent LLM calls (the key agentic pattern) ──────────────────────────

import httpx

async def call_llm(client: httpx.AsyncClient, prompt: str, model: str) -> dict:
    """Single LLM call — async, non-blocking."""
    response = await client.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


async def parallel_llm_calls(prompts: list[str]) -> list[dict | Exception]:
    """
    Fan out N LLM calls concurrently.
    Uses a semaphore to avoid hammering the API.
    Returns results in same order as prompts.
    """
    semaphore = asyncio.Semaphore(5)  # max 5 concurrent requests

    async def bounded_call(client: httpx.AsyncClient, prompt: str) -> dict:
        async with semaphore:
            return await call_llm(client, prompt, "llama3-8b-8192")

    async with httpx.AsyncClient() as client:
        tasks = [bounded_call(client, p) for p in prompts]
        # return_exceptions=True prevents one failure killing all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return list(results)


# ─── Endpoints demonstrating these patterns ───────────────────────────────────

@router.get("/users/{user_id}")
async def get_user(
    user_id: UUID,
    service: UserServiceDep,          # injected automatically
):
    user = await service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.post("/agents/parallel-analyze")
async def parallel_analyze(
    texts: list[str],
    background_tasks: BackgroundTasks,
    service: UserServiceDep,
):
    """
    Analyze multiple texts concurrently via LLM.
    Background task logs results after response is sent.
    """
    if len(texts) > 10:
        raise HTTPException(status_code=400, detail="Max 10 texts per request")

    prompts = [f"Analyze this text in one sentence: {t}" for t in texts]
    results = await parallel_llm_calls(prompts)

    # Process results — separate successes from failures
    processed = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed.append({"text": texts[i], "error": str(result), "success": False})
        else:
            content = result["choices"][0]["message"]["content"]
            processed.append({"text": texts[i], "analysis": content, "success": True})

    # Log to DB asynchronously AFTER response is sent
    background_tasks.add_task(log_analysis_results, processed)

    return {"results": processed, "total": len(texts)}


async def log_analysis_results(results: list[dict]) -> None:
    """Runs in background after response is sent."""
    await asyncio.sleep(0)  # yield control
    success_count = sum(1 for r in results if r["success"])
    print(f"Logged {success_count}/{len(results)} successful analyses")


# ─── Python 3.11+ TaskGroup (safer than gather) ───────────────────────────────

async def run_agent_tasks(agent_inputs: list[dict]) -> list[dict]:
    """
    TaskGroup cancels ALL tasks if any raises.
    Use this when all results are required (fail-fast).
    Use gather(return_exceptions=True) when partial results are OK.
    """
    results = []
    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(process_agent_input(inp))
            for inp in agent_inputs
        ]
    results = [t.result() for t in tasks]
    return results


async def process_agent_input(inp: dict) -> dict:
    await asyncio.sleep(0.1)  # simulate async work
    return {"processed": True, **inp}
