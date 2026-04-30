"""
================================================================================
MODULE 06 — Performance, Caching & Background Tasks
================================================================================
Topics:
  L1. Redis caching — cache-aside & write-through
  L2. Celery for long-running AI tasks
  L3. Server-Sent Events for LLM token streaming
  L4. Rate limiting — per-user sliding window
  L5. Async HTTP clients & connection pools

Run:  uvicorn main:app --reload
      Redis optional (app gracefully degrades without it)
Docs: http://localhost:8000/docs
================================================================================
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import uuid4

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 1 — Redis Cache (simulated without real Redis dependency)
# ─────────────────────────────────────────────────────────────────────────────
# In production, replace InMemoryCache with:
#   import redis.asyncio as redis
#   client = redis.from_url("redis://localhost:6379", decode_responses=True)

class InMemoryCache:
    """
    Simulates Redis for local development.
    Interface matches redis.asyncio for easy swap.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._ttls: dict[str, float] = {}  # key → expiry timestamp

    async def get(self, key: str) -> str | None:
        if key not in self._store:
            return None
        if key in self._ttls and time.time() > self._ttls[key]:
            del self._store[key]
            del self._ttls[key]
            return None
        return self._store[key]

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._store[key] = value
        if ex:
            self._ttls[key] = time.time() + ex

    async def delete(self, key: str) -> int:
        existed = key in self._store
        self._store.pop(key, None)
        self._ttls.pop(key, None)
        return 1 if existed else 0

    async def exists(self, *keys: str) -> int:
        return sum(1 for k in keys if k in self._store)

    async def incr(self, key: str) -> int:
        val = int(self._store.get(key, 0)) + 1
        self._store[key] = str(val)
        return val

    async def expire(self, key: str, seconds: int) -> None:
        self._ttls[key] = time.time() + seconds


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — shared resources
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Shared async HTTP client — ONE connection pool for the entire app
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    cache = InMemoryCache()

    app.state.http_client = http_client
    app.state.cache = cache
    app.state.tasks: dict[str, dict] = {}  # simulates Celery task results

    print("✅ App resources initialized")
    yield

    await http_client.aclose()
    print("✅ HTTP client closed")


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

def get_cache(request: Request) -> InMemoryCache:
    return request.app.state.cache

def get_http_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client

def get_task_store(request: Request) -> dict:
    return request.app.state.tasks


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 1 — Cache-Aside Pattern
# ─────────────────────────────────────────────────────────────────────────────
# Cache-aside: your code checks cache → miss → fetch from DB → store in cache.
# The application manages the cache, not the DB. Most common pattern.

class ProductCache:
    """
    Wraps the cache client with product-specific logic.
    Serializes Pydantic models to JSON for Redis storage.
    """

    def __init__(self, cache: InMemoryCache) -> None:
        self.cache = cache
        self.TTL = 300  # 5 minutes

    def _key(self, product_id: str) -> str:
        return f"product:v1:{product_id}"  # versioned keys — easy cache busting

    async def get(self, product_id: str) -> dict | None:
        raw = await self.cache.get(self._key(product_id))
        if raw:
            return json.loads(raw)
        return None

    async def set(self, product: dict) -> None:
        await self.cache.set(
            self._key(product["id"]),
            json.dumps(product),
            ex=self.TTL,
        )

    async def invalidate(self, product_id: str) -> None:
        """Call this when a product is updated or deleted."""
        await self.cache.delete(self._key(product_id))


def get_product_cache(cache: InMemoryCache = Depends(get_cache)) -> ProductCache:
    return ProductCache(cache)


# Simulated DB
_products_db: dict[str, dict] = {
    "prod-001": {"id": "prod-001", "name": "Laptop Pro", "price": 999.99},
    "prod-002": {"id": "prod-002", "name": "Wireless Mouse", "price": 29.99},
}

async def fetch_product_from_db(product_id: str) -> dict | None:
    """Simulates a slow DB query (50ms)."""
    await asyncio.sleep(0.05)
    return _products_db.get(product_id)


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 4 — Sliding Window Rate Limiter
# ─────────────────────────────────────────────────────────────────────────────
# Sliding window: count requests in the last N seconds using Redis sorted sets.
# More accurate than fixed windows. No burst at window boundaries.
#
# Production Redis implementation:
#   async def sliding_window_rate_limit(key, limit, window_seconds, redis):
#       now = time.time()
#       pipe = redis.pipeline()
#       pipe.zremrangebyscore(key, 0, now - window_seconds)
#       pipe.zcard(key)
#       pipe.zadd(key, {str(uuid4()): now})
#       pipe.expire(key, window_seconds)
#       _, count, _, _ = await pipe.execute()
#       return count <= limit

_request_counts: dict[str, list[float]] = defaultdict(list)  # ip → [timestamps]

class RateLimiter:
    """Sliding window rate limiter using in-memory lists (Redis in production)."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window = window_seconds

    async def check(self, identifier: str) -> None:
        now = time.time()
        window_start = now - self.window

        # Remove timestamps outside the window
        counts = _request_counts[identifier]
        _request_counts[identifier] = [t for t in counts if t > window_start]

        if len(_request_counts[identifier]) >= self.limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit: {self.limit} requests per {self.window}s",
                headers={"Retry-After": str(self.window)},
            )

        _request_counts[identifier].append(now)


# Rate limiter instances (reuse across requests)
api_limiter = RateLimiter(limit=10, window_seconds=60)   # 10/min for general API
ai_limiter = RateLimiter(limit=5, window_seconds=60)     # 5/min for expensive AI calls


def get_rate_limit_key(request: Request) -> str:
    """Use IP + user agent as the rate limit key."""
    return f"{request.client.host}:{request.headers.get('user-agent', 'unknown')[:50]}"


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 2 — Background Tasks (Celery simulation)
# ─────────────────────────────────────────────────────────────────────────────
# In production: replace with Celery + Redis broker.
# Pattern: route returns task_id immediately, client polls /tasks/{id} for status.

class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending | running | completed | failed
    result: Any = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


async def run_ai_analysis_task(task_id: str, payload: dict, task_store: dict) -> None:
    """
    Simulates a long-running AI task.
    In production: this is a Celery task decorated with @celery.task
    """
    task_store[task_id]["status"] = "running"

    try:
        # Simulate AI processing time (3 seconds)
        await asyncio.sleep(3)

        # Simulate AI result
        result = {
            "analysis": f"Analysis of: {payload.get('text', '')[:50]}",
            "sentiment": "positive",
            "confidence": 0.92,
            "tokens_used": 150,
        }

        task_store[task_id].update({
            "status": "completed",
            "result": result,
            "completed_at": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        task_store[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat(),
        })


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 3 — SSE / Streaming Response
# ─────────────────────────────────────────────────────────────────────────────
# StreamingResponse with an async generator streams tokens as they arrive.
# The client receives data immediately — no waiting for full response.

async def fake_llm_token_stream(prompt: str) -> AsyncGenerator[str, None]:
    """
    Simulates streaming LLM output token-by-token.
    In production: use groq_client.chat.completions.create(stream=True)
    """
    words = f"This is a streaming response to your prompt: '{prompt}'. Each word arrives as a separate token. This is how real LLM streaming works in production systems.".split()

    for i, word in enumerate(words):
        # SSE format: "data: {json}\n\n"
        chunk = {
            "delta": word + " ",
            "index": i,
            "finish_reason": None,
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.05)  # simulate token generation delay

    # Final chunk signals completion
    yield f"data: {json.dumps({'delta': '', 'index': len(words), 'finish_reason': 'stop'})}\n\n"
    yield "data: [DONE]\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# APP & ROUTES
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Module 06 — Performance & Caching",
    description="Redis caching, background tasks, SSE streaming, rate limiting",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/products/{product_id}", tags=["Products"])
async def get_product(
    product_id: str,
    request: Request,
    product_cache: ProductCache = Depends(get_product_cache),
) -> dict[str, Any]:
    """
    LESSON 1 — Cache-aside pattern in action.
    First request: fetches from DB (~50ms), caches the result.
    Second request: returns from cache (~1ms).
    Watch X-Cache-Status header in the response.
    """
    await api_limiter.check(get_rate_limit_key(request))

    cached = await product_cache.get(product_id)
    if cached:
        return {**cached, "_cache": "HIT"}

    product = await fetch_product_from_db(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    await product_cache.set(product)
    return {**product, "_cache": "MISS"}


@app.put("/products/{product_id}", tags=["Products"])
async def update_product(
    product_id: str,
    data: dict,
    product_cache: ProductCache = Depends(get_product_cache),
) -> dict[str, Any]:
    """
    LESSON 1 — Cache invalidation on write.
    Always invalidate the cache when data changes.
    """
    if product_id not in _products_db:
        raise HTTPException(status_code=404, detail="Product not found")

    _products_db[product_id].update(data)
    await product_cache.invalidate(product_id)  # bust the cache

    return {"message": "Updated", "product": _products_db[product_id]}


class AnalysisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.post("/analyze", status_code=202, tags=["AI Tasks"])
async def start_analysis(
    data: AnalysisRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    task_store: dict = Depends(get_task_store),
) -> dict[str, str]:
    """
    LESSON 2 — Fire-and-forget async task.
    Returns 202 Accepted immediately with a task_id.
    Client polls GET /tasks/{task_id} for the result.

    In production: replace BackgroundTasks with:
        task = analyze_text.delay(data.text)
        return {"task_id": task.id}
    """
    await ai_limiter.check(get_rate_limit_key(request))

    task_id = str(uuid4())
    task_store[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "result": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }

    # Schedule the task to run in the background
    # BackgroundTasks runs AFTER the response is sent — non-blocking
    background_tasks.add_task(run_ai_analysis_task, task_id, data.model_dump(), task_store)

    return {"task_id": task_id, "status": "pending", "poll_url": f"/tasks/{task_id}"}


@app.get("/tasks/{task_id}", response_model=TaskStatus, tags=["AI Tasks"])
async def get_task_status(
    task_id: str,
    task_store: dict = Depends(get_task_store),
) -> TaskStatus:
    """
    LESSON 2 — Poll for task completion.
    Returns: pending → running → completed | failed
    """
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatus(**task)


class StreamRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)


@app.post("/stream/chat", tags=["Streaming"])
async def stream_chat(data: StreamRequest) -> StreamingResponse:
    """
    LESSON 3 — Server-Sent Events (SSE) for LLM token streaming.

    Test with curl:
      curl -X POST http://localhost:8000/stream/chat \\
           -H "Content-Type: application/json" \\
           -d '{"prompt": "Tell me about FastAPI"}' \\
           --no-buffer

    Or open: http://localhost:8000/docs and use the form.
    Watch tokens arrive one-by-one in the response.
    """
    return StreamingResponse(
        fake_llm_token_stream(data.prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disables Nginx buffering
        },
    )


@app.get("/stream/chat-ui", tags=["Streaming"])
async def stream_chat_ui() -> StreamingResponse:
    """Simple HTML page to visualize SSE streaming in a browser."""
    html = """
    <!DOCTYPE html>
    <html>
    <body style="font-family:monospace;padding:20px;background:#0d1117;color:#c9d1d9">
    <h2>LLM Token Stream Demo</h2>
    <input id="prompt" value="Tell me about agentic AI systems" style="width:400px;padding:8px">
    <button onclick="startStream()" style="padding:8px 16px;margin-left:8px">Stream</button>
    <pre id="output" style="background:#161b22;padding:16px;border-radius:8px;min-height:100px;margin-top:16px"></pre>
    <script>
    async function startStream() {
        const prompt = document.getElementById('prompt').value;
        document.getElementById('output').textContent = '';
        const resp = await fetch('/stream/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({prompt})
        });
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            const text = decoder.decode(value);
            for (const line of text.split('\\n')) {
                if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                    const chunk = JSON.parse(line.slice(6));
                    document.getElementById('output').textContent += chunk.delta;
                }
            }
        }
    }
    </script>
    </body></html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)


@app.get("/external/joke", tags=["HTTP Client"])
async def fetch_joke(
    client: httpx.AsyncClient = Depends(get_http_client),
) -> dict[str, Any]:
    """
    LESSON 5 — Shared async HTTP client from lifespan.
    The same connection pool is reused across all requests — efficient.
    """
    try:
        response = await client.get("https://official-joke-api.appspot.com/random_joke")
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="External API timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"External API error: {e.response.status_code}")


# ─────────────────────────────────────────────────────────────────────────────
# CELERY SETUP (Lesson 2 — production version)
# ─────────────────────────────────────────────────────────────────────────────
# pip install celery redis
#
# celery_app.py:
#   from celery import Celery
#   celery = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")
#
#   @celery.task(bind=True, max_retries=3, default_retry_delay=60)
#   def analyze_text(self, text: str) -> dict:
#       try:
#           result = llm_client.analyze(text)
#           return result
#       except Exception as exc:
#           raise self.retry(exc=exc)
#
# In FastAPI route:
#   from celery_app import analyze_text
#   task = analyze_text.delay(data.text)
#   return {"task_id": task.id}
#
# Start worker:
#   celery -A celery_app worker --loglevel=info

# ─────────────────────────────────────────────────────────────────────────────
# PRACTICE EXERCISES
# ─────────────────────────────────────────────────────────────────────────────
# 1. Implement cache stampede prevention using a lock (asyncio.Lock per key).
# 2. Add a cache stats endpoint: hits, misses, hit_rate.
# 3. Make the rate limiter return a Retry-After header with exact wait time.
# 4. Add task cancellation: DELETE /tasks/{task_id} cancels pending tasks.
# 5. Implement write-through caching: update DB and cache atomically.
