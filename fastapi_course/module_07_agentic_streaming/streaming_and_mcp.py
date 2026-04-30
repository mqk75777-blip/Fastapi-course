"""
MODULE 7: Streaming, WebSockets & Agentic Endpoints
=====================================================
- SSE streaming for LLM token-by-token output
- WebSocket bidirectional agent communication
- Tool call API (agents invoke your FastAPI as a tool)
- MCP (Model Context Protocol) server on FastAPI
- Long-running task endpoints with status polling
"""

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import settings

router = APIRouter()


# ─── 1. SSE Streaming — LLM tokens to frontend ───────────────────────────────

async def stream_groq_response(messages: list[dict], model: str) -> AsyncGenerator[str, None]:
    """
    Streams tokens from Groq (or any OpenAI-compatible API) as SSE events.
    Each yielded string is a complete SSE event.
    """
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "max_tokens": 1024,
            }
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    yield "data: [DONE]\n\n"
                    return
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    if token := delta.get("content"):
                        # Forward SSE event to client
                        event_data = json.dumps({"token": token, "type": "token"})
                        yield f"data: {event_data}\n\n"
                except (json.JSONDecodeError, KeyError):
                    continue


class ChatRequest(BaseModel):
    messages: list[dict]
    model: str = "llama3-8b-8192"
    session_id: str | None = None


@router.post("/chat/stream")
async def stream_chat(body: ChatRequest):
    """
    SSE endpoint — frontend connects and receives tokens as they arrive.
    Usage: EventSource('/api/v1/chat/stream') doesn't work for POST.
    Use fetch() with ReadableStream on the frontend instead.
    """
    return StreamingResponse(
        stream_groq_response(body.messages, body.model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable Nginx buffering
            "Connection": "keep-alive",
        }
    )


# ─── 2. WebSocket — bidirectional agent communication ────────────────────────

class ConnectionManager:
    """Manages WebSocket connections for multi-user scenarios."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str) -> None:
        self.active_connections.pop(client_id, None)

    async def send_to(self, client_id: str, data: dict) -> None:
        if ws := self.active_connections.get(client_id):
            await ws.send_json(data)

    async def broadcast(self, data: dict) -> None:
        for ws in self.active_connections.values():
            await ws.send_json(data)


manager = ConnectionManager()


@router.websocket("/ws/{client_id}")
async def websocket_agent(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time agent interaction.
    Client sends: {"type": "message", "content": "..."}
    Server streams back tokens and tool calls in real time.
    """
    await manager.connect(websocket, client_id)
    conversation_history: list[dict] = []

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            if event.get("type") == "message":
                user_content = event.get("content", "")
                conversation_history.append({"role": "user", "content": user_content})

                # Acknowledge receipt immediately
                await websocket.send_json({"type": "start", "message_id": str(uuid.uuid4())})

                # Stream LLM response token by token
                full_response = ""
                async with httpx.AsyncClient(timeout=60) as client:
                    async with client.stream(
                        "POST",
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                        json={
                            "model": "llama3-8b-8192",
                            "messages": conversation_history,
                            "stream": True,
                        }
                    ) as response:
                        async for line in response.aiter_lines():
                            if not line.startswith("data: ") or line[6:] == "[DONE]":
                                continue
                            try:
                                chunk = json.loads(line[6:])
                                token = chunk["choices"][0]["delta"].get("content", "")
                                if token:
                                    full_response += token
                                    await websocket.send_json({"type": "token", "token": token})
                            except (json.JSONDecodeError, KeyError):
                                continue

                conversation_history.append({"role": "assistant", "content": full_response})
                await websocket.send_json({"type": "done", "full_response": full_response})

            elif event.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(client_id)


# ─── 3. Tool call API (your FastAPI as an agent tool) ─────────────────────────

class ToolCallRequest(BaseModel):
    """
    Standard schema for tool calls from LLM agents.
    Compatible with OpenAI function calling and Anthropic tool use.
    """
    tool_name: str
    arguments: dict[str, Any]
    call_id: str = ""
    agent_id: str | None = None


class ToolCallResponse(BaseModel):
    tool_name: str
    call_id: str
    result: Any
    error: str | None = None
    execution_time_ms: float


REGISTERED_TOOLS: dict[str, Any] = {}


def register_tool(name: str):
    """Decorator to register a function as a callable tool for agents."""
    def decorator(func):
        REGISTERED_TOOLS[name] = func
        return func
    return decorator


@register_tool("search_knowledge_base")
async def search_knowledge_base(query: str, limit: int = 5) -> dict:
    """Search vector DB for relevant documents."""
    # Real implementation uses EmbeddingRepository from Module 5
    return {"results": [f"Mock result for: {query}"], "count": 1}


@register_tool("get_weather")
async def get_weather(city: str, country: str = "PK") -> dict:
    return {"city": city, "temperature": 28, "condition": "sunny"}


@register_tool("send_email")
async def send_email(to: str, subject: str, body: str) -> dict:
    # Integrate with SMTP / SendGrid here
    return {"sent": True, "to": to}


@router.post("/tools/call", response_model=ToolCallResponse)
async def execute_tool(body: ToolCallRequest):
    """
    LLM agents call this endpoint to execute tools.
    Register tools with @register_tool decorator above.
    """
    import time
    start = time.time()

    tool_fn = REGISTERED_TOOLS.get(body.tool_name)
    if not tool_fn:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{body.tool_name}' not found. Available: {list(REGISTERED_TOOLS.keys())}"
        )

    try:
        result = await tool_fn(**body.arguments)
        return ToolCallResponse(
            tool_name=body.tool_name,
            call_id=body.call_id,
            result=result,
            execution_time_ms=round((time.time() - start) * 1000, 2),
        )
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid arguments: {e}")
    except Exception as e:
        return ToolCallResponse(
            tool_name=body.tool_name,
            call_id=body.call_id,
            result=None,
            error=str(e),
            execution_time_ms=round((time.time() - start) * 1000, 2),
        )


@router.get("/tools/list")
async def list_tools():
    """Returns available tools with their schemas — for agent discovery."""
    import inspect
    tools = []
    for name, fn in REGISTERED_TOOLS.items():
        sig = inspect.signature(fn)
        tools.append({
            "name": name,
            "description": fn.__doc__ or "",
            "parameters": {
                k: {"type": str(v.annotation.__name__ if v.annotation != inspect.Parameter.empty else "any")}
                for k, v in sig.parameters.items()
            }
        })
    return {"tools": tools}


# ─── 4. MCP Server on FastAPI ─────────────────────────────────────────────────
# Model Context Protocol — makes your FastAPI a tool server for Claude/LLM clients

mcp_router = APIRouter(prefix="/mcp", tags=["mcp"])


@mcp_router.get("/")
async def mcp_info():
    """MCP server discovery endpoint."""
    return {
        "protocol_version": "2024-11-05",
        "server_info": {"name": "FastAPI MCP Server", "version": "1.0.0"},
        "capabilities": {"tools": {}},
    }


@mcp_router.post("/tools/list")
async def mcp_list_tools():
    """MCP tools/list — returns tool schemas in MCP format."""
    return {
        "tools": [
            {
                "name": "search_knowledge_base",
                "description": "Search the knowledge base for relevant information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "country": {"type": "string", "default": "PK"}
                    },
                    "required": ["city"]
                }
            }
        ]
    }


@mcp_router.post("/tools/call")
async def mcp_call_tool(body: dict):
    """MCP tools/call — executes a tool and returns result in MCP format."""
    tool_name = body.get("name")
    arguments = body.get("arguments", {})

    tool_fn = REGISTERED_TOOLS.get(tool_name)
    if not tool_fn:
        return {
            "content": [{"type": "text", "text": f"Tool '{tool_name}' not found"}],
            "isError": True,
        }

    try:
        result = await tool_fn(**arguments)
        return {
            "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
            "isError": False,
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error: {str(e)}"}],
            "isError": True,
        }


# ─── 5. Long-running tasks with polling ───────────────────────────────────────

task_store: dict[str, dict] = {}  # Use Redis in production (Module 5)


class AgentTaskRequest(BaseModel):
    task_type: str
    payload: dict
    webhook_url: str | None = None


@router.post("/tasks/submit")
async def submit_task(body: AgentTaskRequest, background_tasks: BackgroundTasks):
    """Submit a long-running task. Returns task_id for polling."""
    task_id = str(uuid.uuid4())
    task_store[task_id] = {
        "id": task_id,
        "status": "queued",
        "task_type": body.task_type,
        "created_at": datetime.utcnow().isoformat(),
        "result": None,
        "error": None,
    }
    background_tasks.add_task(process_agent_task, task_id, body)
    return {"task_id": task_id, "status": "queued", "poll_url": f"/api/v1/tasks/{task_id}"}


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


async def process_agent_task(task_id: str, body: AgentTaskRequest) -> None:
    task_store[task_id]["status"] = "running"
    try:
        await asyncio.sleep(3)  # simulate work
        result = {"processed": True, "task_type": body.task_type, "output": "Done"}
        task_store[task_id].update({"status": "completed", "result": result})

        # Send webhook if provided
        if body.webhook_url:
            async with httpx.AsyncClient() as client:
                await client.post(body.webhook_url, json=task_store[task_id])

    except Exception as e:
        task_store[task_id].update({"status": "failed", "error": str(e)})
