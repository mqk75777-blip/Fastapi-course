"""
================================================================================
MODULE 10 — Building Production Agentic AI APIs (CAPSTONE)
================================================================================
Topics:
  L1. Tool registry — dynamic tool discovery API
  L2. Agent memory API with vector search (simulated)
  L3. Multi-agent orchestration endpoints
  L4. Human-in-the-loop approval API
  L5. OpenAI-compatible /v1/chat/completions endpoint (Groq backend)
  L6. CAPSTONE — full production system

Run:  uvicorn main:app --reload
      Set GROQ_API_KEY env var for real LLM calls (L5)
      Or leave unset — falls back to simulation mode
Docs: http://localhost:8000/docs
================================================================================
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 1 — Tool Registry
# ─────────────────────────────────────────────────────────────────────────────
# A tool registry lets agents discover what tools are available,
# understand their schemas, and call them dynamically.
# This is the server-side equivalent of a function/tool definition in LLM APIs.

class ToolParameter(BaseModel):
    type: str                       # "string", "integer", "boolean", "object", "array"
    description: str
    required: bool = False
    enum: list[str] | None = None   # for string parameters with fixed choices
    default: Any = None


class ToolSchema(BaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[a-z_][a-z0-9_]*$")
    description: str = Field(min_length=10, max_length=500)
    version: str = "1.0.0"
    parameters: dict[str, ToolParameter]
    required_parameters: list[str] = Field(default_factory=list)
    returns: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    is_active: bool = True
    rate_limit_per_minute: int = 60


class ToolRegistration(BaseModel):
    schema: ToolSchema
    endpoint_url: str = Field(description="URL where this tool is callable")
    auth_required: bool = False


class ToolCall(BaseModel):
    tool_name: str
    parameters: dict[str, Any]
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ToolResult(BaseModel):
    call_id: str
    tool_name: str
    result: Any
    error: str | None = None
    execution_time_ms: float
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# Tool registry store
_tool_registry: dict[str, ToolRegistration] = {}

# Built-in tools
def _register_builtin_tools() -> None:
    tools = [
        ToolRegistration(
            schema=ToolSchema(
                name="web_search",
                description="Search the web for current information on any topic",
                parameters={
                    "query": ToolParameter(type="string", description="The search query", required=True),
                    "num_results": ToolParameter(type="integer", description="Number of results (1-10)", default=5),
                },
                required_parameters=["query"],
                returns={"results": "array of {title, url, snippet}"},
                tags=["search", "web"],
            ),
            endpoint_url="/tools/execute/web_search",
        ),
        ToolRegistration(
            schema=ToolSchema(
                name="calculator",
                description="Perform mathematical calculations. Supports basic arithmetic and common functions.",
                parameters={
                    "expression": ToolParameter(type="string", description="Math expression to evaluate, e.g. '2 + 2 * 10'", required=True),
                },
                required_parameters=["expression"],
                returns={"result": "number"},
                tags=["math", "utility"],
            ),
            endpoint_url="/tools/execute/calculator",
        ),
        ToolRegistration(
            schema=ToolSchema(
                name="memory_search",
                description="Search the agent's long-term memory for relevant past information",
                parameters={
                    "query": ToolParameter(type="string", description="What to search for in memory", required=True),
                    "limit": ToolParameter(type="integer", description="Max results (1-20)", default=5),
                },
                required_parameters=["query"],
                returns={"memories": "array of {content, relevance_score, timestamp}"},
                tags=["memory", "retrieval"],
            ),
            endpoint_url="/tools/execute/memory_search",
        ),
    ]
    for tool in tools:
        _tool_registry[tool.schema.name] = tool

_register_builtin_tools()


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 2 — Agent Memory API (Vector Search Simulated)
# ─────────────────────────────────────────────────────────────────────────────
# In production: use pgvector (PostgreSQL extension) or Qdrant/Pinecone.
# Store embeddings alongside text. Search by cosine similarity.
# Here we simulate embeddings with character frequency vectors.

class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    content: str
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class MemoryCreate(BaseModel):
    agent_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchResult(BaseModel):
    memory: Memory
    relevance_score: float


_memories: list[Memory] = []


def _fake_embed(text: str) -> list[float]:
    """
    Fake embedding: 26-dim vector of character frequencies.
    In production: call an embedding model API (OpenAI, Groq, etc.)
    e.g., client.embeddings.create(input=text, model="text-embedding-3-small")
    """
    vec = [0.0] * 26
    text_lower = text.lower()
    total = max(len(text_lower), 1)
    for char in text_lower:
        if char.isalpha():
            vec[ord(char) - ord('a')] += 1
    # Normalize to unit vector
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Range: [-1, 1]. 1 = identical."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 4 — Human-in-the-Loop (HITL) Approval
# ─────────────────────────────────────────────────────────────────────────────
# Agents sometimes need human approval before taking sensitive actions:
# - Sending emails
# - Making purchases
# - Deleting data
# - Calling external APIs with real consequences
#
# Pattern:
#   1. Agent requests approval → status: pending
#   2. Human reviews and approves/rejects via API
#   3. Agent polls or receives webhook → proceeds or aborts

class ApprovalStatus(str):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    action_type: str                    # "send_email", "delete_records", "api_call"
    action_description: str
    action_payload: dict[str, Any]      # the full action to take if approved
    risk_level: Literal["low", "medium", "high", "critical"]
    requested_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: str                     # auto-reject after this time
    status: str = "pending"
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    review_note: str | None = None


class ApprovalResponse(BaseModel):
    action: Literal["approve", "reject"]
    reviewer_id: str
    note: str | None = None


_approvals: dict[str, ApprovalRequest] = {}


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 3 — Multi-Agent Orchestration
# ─────────────────────────────────────────────────────────────────────────────

class AgentTask(BaseModel):
    agent_id: str
    task: str
    tools: list[str] = Field(default_factory=list)


class OrchestrationRequest(BaseModel):
    orchestrator_id: str
    goal: str
    sub_agents: list[AgentTask]
    parallel: bool = True   # run agents in parallel vs sequential


class AgentTaskResult(BaseModel):
    agent_id: str
    task: str
    result: str
    tool_calls_made: int
    execution_time_ms: float


class OrchestrationResult(BaseModel):
    orchestration_id: str
    goal: str
    agent_results: list[AgentTaskResult]
    synthesized_result: str
    total_time_ms: float


async def run_sub_agent(task: AgentTask) -> AgentTaskResult:
    """
    Simulates a sub-agent completing a task.
    In production: call your agent runner (LangChain, CrewAI, custom, etc.)
    """
    start = time.perf_counter()
    await asyncio.sleep(0.5)  # simulate agent thinking and tool calls
    elapsed_ms = (time.perf_counter() - start) * 1000

    return AgentTaskResult(
        agent_id=task.agent_id,
        task=task.task,
        result=f"Agent {task.agent_id} completed: {task.task[:50]}. Found relevant information and synthesized a response.",
        tool_calls_made=len(task.tools),
        execution_time_ms=round(elapsed_ms, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 5 — OpenAI-Compatible /v1/chat/completions
# ─────────────────────────────────────────────────────────────────────────────
# Many tools and frameworks (LangChain, OpenAI SDK) expect the OpenAI API format.
# By implementing this endpoint, your server becomes a drop-in API backend.
# This lets you: swap providers, add caching, add observability transparently.

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama3-8b-8192"

# Usage tracking (use Redis + TimeSeries in production)
_usage: dict[str, int] = {}


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: list[ChatMessage]
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=1000, ge=1, le=8192)
    stream: bool = False
    # Model aliasing — map friendly names to actual model IDs
    # "fast" → "llama3-8b-8192", "smart" → "llama3-70b-8192"


MODEL_ALIASES: dict[str, str] = {
    "fast": "llama3-8b-8192",
    "smart": "llama3-70b-8192",
    "default": "llama3-8b-8192",
}


async def call_groq_streaming(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    http_client: httpx.AsyncClient,
) -> AsyncGenerator[str, None]:
    """Proxies a streaming request to Groq and forwards SSE chunks."""
    async with http_client.stream(
        "POST",
        f"{GROQ_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        },
        timeout=60.0,
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                yield f"{line}\n\n"


async def fake_streaming_response(messages: list[dict], model: str) -> AsyncGenerator[str, None]:
    """Fallback when no Groq API key is set."""
    words = f"This is a simulated response from model {model}. Set GROQ_API_KEY to use the real API.".split()
    for i, word in enumerate(words):
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4()!s:.8}",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"delta": {"content": word + " "}, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.03)

    final = {
        "id": f"chatcmpl-{uuid.uuid4()!s:.8}",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    http_client = httpx.AsyncClient(timeout=60.0)
    app.state.http_client = http_client
    print(f"✅ Module 10 started. Groq API key: {'set' if GROQ_API_KEY else 'NOT SET (simulation mode)'}")
    yield
    await http_client.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Module 10 — Production Agentic AI APIs",
    description="Tool registry, memory, HITL, multi-agent orchestration, OpenAI-compatible endpoint",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 1 — Tool Registry Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/tools", response_model=list[ToolSchema], tags=["Tool Registry"])
async def list_tools(tag: str | None = None) -> list[ToolSchema]:
    """List all registered tools. Filter by tag for agent-specific discovery."""
    tools = [r.schema for r in _tool_registry.values() if r.schema.is_active]
    if tag:
        tools = [t for t in tools if tag in t.tags]
    return tools


@app.get("/tools/{tool_name}", response_model=ToolRegistration, tags=["Tool Registry"])
async def get_tool(tool_name: str) -> ToolRegistration:
    tool = _tool_registry.get(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    return tool


@app.post("/tools", response_model=ToolRegistration, status_code=201, tags=["Tool Registry"])
async def register_tool(data: ToolRegistration) -> ToolRegistration:
    """Register a new tool. Agents can discover it via GET /tools."""
    if data.schema.name in _tool_registry:
        raise HTTPException(status_code=409, detail=f"Tool '{data.schema.name}' already exists")
    _tool_registry[data.schema.name] = data
    return data


@app.post("/tools/{tool_name}/call", response_model=ToolResult, tags=["Tool Registry"])
async def call_tool(tool_name: str, call: ToolCall) -> ToolResult:
    """Execute a registered tool. Returns structured result for agent consumption."""
    tool = _tool_registry.get(tool_name)
    if not tool or not tool.schema.is_active:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found or inactive")

    # Validate required parameters
    for param_name in tool.schema.required_parameters:
        if param_name not in call.parameters:
            raise HTTPException(status_code=422, detail=f"Missing required parameter: {param_name}")

    start = time.perf_counter()

    # Execute tool (simulated)
    if tool_name == "calculator":
        try:
            expr = call.parameters.get("expression", "")
            # Safe eval — only allow math operations
            allowed = set("0123456789+-*/().,% ")
            if not all(c in allowed for c in expr):
                raise ValueError("Expression contains unsafe characters")
            result = eval(expr)  # noqa: S307 — restricted above
        except Exception as e:
            result = {"error": str(e)}
    elif tool_name == "web_search":
        result = {
            "results": [
                {"title": f"Result {i} for '{call.parameters['query']}'", "url": f"https://example.com/{i}", "snippet": "..."}
                for i in range(1, call.parameters.get("num_results", 3) + 1)
            ]
        }
    elif tool_name == "memory_search":
        query_embedding = _fake_embed(call.parameters.get("query", ""))
        scored = [
            (m, _cosine_similarity(query_embedding, m.embedding))
            for m in _memories
            if m.embedding
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        result = {
            "memories": [
                {"content": m.content, "relevance_score": round(score, 3), "timestamp": m.created_at}
                for m, score in scored[:call.parameters.get("limit", 5)]
            ]
        }
    else:
        result = {"message": f"Tool '{tool_name}' executed with params: {call.parameters}"}

    elapsed_ms = (time.perf_counter() - start) * 1000

    return ToolResult(
        call_id=call.call_id,
        tool_name=tool_name,
        result=result,
        execution_time_ms=round(elapsed_ms, 3),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 2 — Memory API Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/memory", response_model=Memory, status_code=201, tags=["Memory"])
async def store_memory(data: MemoryCreate) -> Memory:
    """Store a memory with auto-generated embedding."""
    memory = Memory(
        agent_id=data.agent_id,
        content=data.content,
        embedding=_fake_embed(data.content),
        metadata=data.metadata,
    )
    _memories.append(memory)
    return memory


@app.post("/memory/search", response_model=list[MemorySearchResult], tags=["Memory"])
async def search_memory(
    agent_id: str,
    query: str,
    limit: int = 5,
) -> list[MemorySearchResult]:
    """Find the most relevant memories using cosine similarity (simulated vector search)."""
    query_embedding = _fake_embed(query)
    agent_memories = [m for m in _memories if m.agent_id == agent_id]

    scored = [
        MemorySearchResult(memory=m, relevance_score=round(_cosine_similarity(query_embedding, m.embedding), 4))
        for m in agent_memories
        if m.embedding
    ]
    scored.sort(key=lambda x: x.relevance_score, reverse=True)
    return scored[:limit]


@app.get("/memory/{agent_id}", response_model=list[Memory], tags=["Memory"])
async def get_memories(agent_id: str, limit: int = 20) -> list[Memory]:
    """Get all memories for an agent, newest first."""
    agent_memories = [m for m in _memories if m.agent_id == agent_id]
    return sorted(agent_memories, key=lambda m: m.created_at, reverse=True)[:limit]


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 3 — Multi-Agent Orchestration Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/orchestrate", response_model=OrchestrationResult, tags=["Orchestration"])
async def orchestrate(data: OrchestrationRequest) -> OrchestrationResult:
    """
    Orchestrate multiple agents towards a shared goal.
    parallel=True: all agents run simultaneously via asyncio.gather (faster).
    parallel=False: agents run sequentially (when each depends on the previous).
    """
    start = time.perf_counter()
    orchestration_id = str(uuid.uuid4())

    if data.parallel:
        # Fan-out: all agents run at the same time
        results = await asyncio.gather(*[run_sub_agent(task) for task in data.sub_agents])
    else:
        # Sequential: each agent's result can be fed to the next
        results = []
        for task in data.sub_agents:
            result = await run_sub_agent(task)
            results.append(result)

    total_ms = (time.perf_counter() - start) * 1000

    # Synthesize results (in production: use an LLM to merge outputs)
    synthesis = f"Goal '{data.goal}' completed by {len(results)} agents. " + " | ".join(
        f"{r.agent_id}: {r.result[:40]}..." for r in results
    )

    return OrchestrationResult(
        orchestration_id=orchestration_id,
        goal=data.goal,
        agent_results=list(results),
        synthesized_result=synthesis,
        total_time_ms=round(total_ms, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 4 — Human-in-the-Loop Routes
# ─────────────────────────────────────────────────────────────────────────────

class ApprovalCreate(BaseModel):
    agent_id: str
    action_type: str
    action_description: str
    action_payload: dict[str, Any]
    risk_level: Literal["low", "medium", "high", "critical"]
    ttl_seconds: int = Field(default=3600, ge=60, le=86400)


@app.post("/approvals", response_model=ApprovalRequest, status_code=202, tags=["HITL"])
async def request_approval(data: ApprovalCreate) -> ApprovalRequest:
    """
    Agent requests human approval before taking a sensitive action.
    Returns 202 — the action has not happened yet.
    Agent should poll GET /approvals/{id} or receive a webhook.
    """
    from datetime import timedelta
    expires = (datetime.utcnow() + timedelta(seconds=data.ttl_seconds)).isoformat()

    approval = ApprovalRequest(
        agent_id=data.agent_id,
        action_type=data.action_type,
        action_description=data.action_description,
        action_payload=data.action_payload,
        risk_level=data.risk_level,
        expires_at=expires,
    )
    _approvals[approval.id] = approval
    return approval


@app.get("/approvals", response_model=list[ApprovalRequest], tags=["HITL"])
async def list_pending_approvals() -> list[ApprovalRequest]:
    """Human reviewer endpoint — see all pending approvals."""
    return [a for a in _approvals.values() if a.status == "pending"]


@app.get("/approvals/{approval_id}", response_model=ApprovalRequest, tags=["HITL"])
async def get_approval(approval_id: str) -> ApprovalRequest:
    """Agent polls this endpoint to check if the action has been approved."""
    approval = _approvals.get(approval_id)
    if not approval:
        raise HTTPException(status_code=404, detail="Approval request not found")

    # Auto-expire
    if approval.status == "pending" and datetime.utcnow().isoformat() > approval.expires_at:
        approval.status = "expired"

    return approval


@app.post("/approvals/{approval_id}/review", response_model=ApprovalRequest, tags=["HITL"])
async def review_approval(approval_id: str, review: ApprovalResponse) -> ApprovalRequest:
    """Human reviewer approves or rejects an action."""
    approval = _approvals.get(approval_id)
    if not approval:
        raise HTTPException(status_code=404, detail="Approval request not found")
    if approval.status != "pending":
        raise HTTPException(status_code=409, detail=f"Approval is already {approval.status}")

    approval.status = "approved" if review.action == "approve" else "rejected"
    approval.reviewed_by = review.reviewer_id
    approval.reviewed_at = datetime.utcnow().isoformat()
    approval.review_note = review.note

    return approval


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 5 — OpenAI-Compatible /v1/chat/completions
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions", tags=["OpenAI Compatible"])
async def chat_completions(
    request: Request,
    data: ChatCompletionRequest,
) -> Any:
    """
    OpenAI-compatible chat completions endpoint.
    Drop-in replacement: point any OpenAI SDK client at this server.

    client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="any")
    response = client.chat.completions.create(model="fast", messages=[...])

    Model aliases:
      "fast"    → llama3-8b-8192
      "smart"   → llama3-70b-8192
      "default" → llama3-8b-8192
    """
    # Resolve model alias
    actual_model = MODEL_ALIASES.get(data.model, data.model)

    # Track usage per model
    _usage[actual_model] = _usage.get(actual_model, 0) + 1

    messages = [m.model_dump(exclude_none=True) for m in data.messages]
    http_client: httpx.AsyncClient = request.app.state.http_client

    if data.stream:
        # Streaming response
        if GROQ_API_KEY:
            stream_gen = call_groq_streaming(
                messages=messages,
                model=actual_model,
                temperature=data.temperature or 0.7,
                max_tokens=data.max_tokens or 1000,
                http_client=http_client,
            )
        else:
            stream_gen = fake_streaming_response(messages, actual_model)

        return StreamingResponse(
            stream_gen,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    else:
        # Non-streaming response
        if GROQ_API_KEY:
            response = await http_client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": actual_model,
                    "messages": messages,
                    "temperature": data.temperature,
                    "max_tokens": data.max_tokens,
                },
            )
            response.raise_for_status()
            return response.json()
        else:
            # Simulation response (no API key)
            return {
                "id": f"chatcmpl-{uuid.uuid4()!s:.8}",
                "object": "chat.completion",
                "model": actual_model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"[SIMULATION] Set GROQ_API_KEY for real responses. You asked: {messages[-1].get('content', '')[:100]}",
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
            }


@app.get("/v1/models", tags=["OpenAI Compatible"])
async def list_models() -> dict[str, Any]:
    """OpenAI-compatible models list endpoint."""
    return {
        "object": "list",
        "data": [
            {"id": "fast", "object": "model", "description": "Fast model (llama3-8b)"},
            {"id": "smart", "object": "model", "description": "Smart model (llama3-70b)"},
            {"id": "default", "object": "model", "description": "Default model"},
        ],
    }


@app.get("/v1/usage", tags=["OpenAI Compatible"])
async def usage_stats() -> dict[str, Any]:
    """Track which models are being used."""
    return {"usage_by_model": _usage, "total_requests": sum(_usage.values())}


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 6 — CAPSTONE: Demo Workflow
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/demo/full-agent-workflow", tags=["Capstone"])
async def full_agent_workflow() -> dict[str, Any]:
    """
    CAPSTONE — Runs a complete agent workflow demonstrating all systems:
    1. Look up available tools
    2. Store a memory
    3. Search memory
    4. Request human approval for a sensitive action
    5. Orchestrate sub-agents
    6. Call the LLM endpoint

    This is the architecture pattern you'll use in real agentic systems.
    """
    results: dict[str, Any] = {}

    # Step 1: Tool discovery
    tools = [r.schema.name for r in _tool_registry.values() if r.schema.is_active]
    results["step_1_tools_available"] = tools

    # Step 2: Store a memory
    memory = Memory(
        agent_id="capstone-agent",
        content="The user prefers concise, technical answers about FastAPI and agentic systems.",
        embedding=_fake_embed("FastAPI agentic technical preferences"),
    )
    _memories.append(memory)
    results["step_2_memory_stored"] = memory.id

    # Step 3: Search memory
    query_emb = _fake_embed("FastAPI preferences")
    scored = sorted(
        [(m, _cosine_similarity(query_emb, m.embedding)) for m in _memories if m.embedding],
        key=lambda x: x[1], reverse=True,
    )
    results["step_3_memory_search"] = {
        "query": "FastAPI preferences",
        "top_result": scored[0][0].content[:80] if scored else None,
        "score": round(scored[0][1], 3) if scored else 0,
    }

    # Step 4: Request approval for a sensitive action
    approval = ApprovalRequest(
        agent_id="capstone-agent",
        action_type="send_email",
        action_description="Send welcome email to new user",
        action_payload={"to": "user@example.com", "subject": "Welcome"},
        risk_level="low",
        expires_at="2099-01-01T00:00:00",
    )
    _approvals[approval.id] = approval
    results["step_4_approval_requested"] = {"approval_id": approval.id, "status": approval.status}

    # Step 5: Multi-agent orchestration
    orch_result = await orchestrate(OrchestrationRequest(
        orchestrator_id="capstone-orchestrator",
        goal="Research FastAPI best practices",
        sub_agents=[
            AgentTask(agent_id="researcher-1", task="Find FastAPI performance tips", tools=["web_search"]),
            AgentTask(agent_id="researcher-2", task="Find FastAPI security best practices", tools=["web_search"]),
        ],
        parallel=True,
    ))
    results["step_5_orchestration"] = {
        "agents_used": len(orch_result.agent_results),
        "total_time_ms": orch_result.total_time_ms,
        "synthesis": orch_result.synthesized_result[:100],
    }

    # Step 6: LLM call (simulation)
    llm_response = await chat_completions(
        request=None,  # type: ignore[arg-type]
        data=ChatCompletionRequest(
            model="default",
            messages=[ChatMessage(role="user", content="Summarize what an agentic FastAPI system does in one sentence.")],
        ),
    )
    if isinstance(llm_response, dict):
        results["step_6_llm_response"] = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")

    results["workflow_complete"] = True
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PRACTICE EXERCISES
# ─────────────────────────────────────────────────────────────────────────────
# 1. Replace _fake_embed with real embeddings using Groq's embedding API.
# 2. Add tool versioning: tools have versions, agents can pin to a version.
# 3. Add webhook support to HITL: POST to a callback URL when approved/rejected.
# 4. Implement tool rate limiting using Redis (LESSON 4 of Module 06).
# 5. Add OpenTelemetry spans to every agent step so you can trace full workflows.
