"""
================================================================================
MODULE 09 — WebSockets & Real-Time Agent APIs
================================================================================
Topics:
  L1. WebSocket fundamentals — lifecycle, auth, disconnect handling
  L2. Streaming LLM agent responses over WebSocket
  L3. WebSocket connection manager — multi-client broadcasting
  L4. Long-running agent sessions with state persistence

Run:  uvicorn main:app --reload
Test: open http://localhost:8000/ws-demo in browser (built-in HTML test client)
Docs: http://localhost:8000/docs
================================================================================
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 3 — WebSocket Connection Manager
# ─────────────────────────────────────────────────────────────────────────────
# Manages all active WebSocket connections.
# In production with multiple instances: use Redis pub/sub to broadcast
# across servers — a message to instance A reaches clients on instance B.

class ConnectionManager:
    """
    Manages WebSocket connections per room/session.
    Thread-safe via asyncio — no locks needed (single-threaded event loop).
    """

    def __init__(self) -> None:
        # room_id → list of active connections
        self._connections: dict[str, list[WebSocket]] = {}
        # client_id → WebSocket for direct messaging
        self._clients: dict[str, WebSocket] = {}
        # connection metadata
        self._metadata: dict[str, dict[str, Any]] = {}

    async def connect(
        self,
        websocket: WebSocket,
        room_id: str,
        client_id: str,
        user_id: str | None = None,
    ) -> None:
        """Accept and register a new connection."""
        await websocket.accept()

        if room_id not in self._connections:
            self._connections[room_id] = []

        self._connections[room_id].append(websocket)
        self._clients[client_id] = websocket
        self._metadata[client_id] = {
            "room_id": room_id,
            "user_id": user_id,
            "connected_at": datetime.utcnow().isoformat(),
        }

        print(f"[WS] Client {client_id} joined room {room_id}. Room size: {len(self._connections[room_id])}")

    def disconnect(self, room_id: str, client_id: str) -> None:
        """Remove a disconnected client."""
        websocket = self._clients.pop(client_id, None)
        self._metadata.pop(client_id, None)

        if room_id in self._connections and websocket:
            try:
                self._connections[room_id].remove(websocket)
            except ValueError:
                pass

        if not self._connections.get(room_id):
            self._connections.pop(room_id, None)

        print(f"[WS] Client {client_id} left room {room_id}")

    async def send_to_client(self, client_id: str, message: dict) -> bool:
        """Send a message to a specific client."""
        ws = self._clients.get(client_id)
        if not ws:
            return False
        try:
            await ws.send_json(message)
            return True
        except Exception:
            return False

    async def broadcast_to_room(self, room_id: str, message: dict, exclude: str | None = None) -> int:
        """
        Broadcast a message to all clients in a room.
        Returns the number of clients that received the message.
        In production: publish to Redis channel, let all server instances subscribe.
        """
        connections = self._connections.get(room_id, [])
        sent = 0
        dead_connections = []

        for ws in connections:
            client_id = self._get_client_id(ws)
            if client_id == exclude:
                continue
            try:
                await ws.send_json(message)
                sent += 1
            except Exception:
                dead_connections.append(ws)

        # Clean up dead connections
        for ws in dead_connections:
            cid = self._get_client_id(ws)
            if cid:
                self.disconnect(room_id, cid)

        return sent

    def _get_client_id(self, ws: WebSocket) -> str | None:
        for cid, stored_ws in self._clients.items():
            if stored_ws is ws:
                return cid
        return None

    def room_info(self, room_id: str) -> dict[str, Any]:
        return {
            "room_id": room_id,
            "connection_count": len(self._connections.get(room_id, [])),
        }

    @property
    def total_connections(self) -> int:
        return len(self._clients)


manager = ConnectionManager()


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 4 — Agent Session State
# ─────────────────────────────────────────────────────────────────────────────
# Agent sessions persist across WebSocket reconnections.
# In production: store in Redis with TTL.

class AgentSession(BaseModel):
    session_id: str
    user_id: str
    agent_id: str
    messages: list[dict[str, Any]] = []
    status: str = "active"   # active | paused | completed
    created_at: str
    last_activity: str
    metadata: dict[str, Any] = {}


_sessions: dict[str, AgentSession] = {}


def create_session(user_id: str, agent_id: str) -> AgentSession:
    now = datetime.utcnow().isoformat()
    session = AgentSession(
        session_id=str(uuid.uuid4()),
        user_id=user_id,
        agent_id=agent_id,
        created_at=now,
        last_activity=now,
    )
    _sessions[session.session_id] = session
    return session


def get_session(session_id: str) -> AgentSession | None:
    return _sessions.get(session_id)


def append_to_session(session_id: str, message: dict) -> None:
    session = _sessions.get(session_id)
    if session:
        session.messages.append(message)
        session.last_activity = datetime.utcnow().isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 2 — Simulated LLM Streaming
# ─────────────────────────────────────────────────────────────────────────────

async def stream_agent_response(
    user_message: str,
    session_id: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Simulates an agent that: thinks → calls a tool → streams final answer.
    In production: replace with Groq/Anthropic streaming API.
    """
    # Phase 1: Thinking
    yield {"type": "thinking", "content": "Analyzing the user's request...", "session_id": session_id}
    await asyncio.sleep(0.3)

    yield {"type": "thinking", "content": "Deciding whether to use a tool...", "session_id": session_id}
    await asyncio.sleep(0.3)

    # Phase 2: Tool call (if needed)
    if "search" in user_message.lower() or "find" in user_message.lower():
        tool_call_id = str(uuid.uuid4())[:8]
        yield {
            "type": "tool_call",
            "tool_name": "search",
            "tool_input": {"query": user_message},
            "call_id": tool_call_id,
            "session_id": session_id,
        }
        await asyncio.sleep(0.5)  # simulate tool execution

        yield {
            "type": "tool_result",
            "call_id": tool_call_id,
            "result": {"items": ["Result 1", "Result 2", "Result 3"]},
            "session_id": session_id,
        }
        await asyncio.sleep(0.2)

    # Phase 3: Streaming final answer token by token
    response_text = f"Based on your request '{user_message}', here is my analysis and response with detailed information."
    words = response_text.split()

    for i, word in enumerate(words):
        yield {
            "type": "token",
            "delta": word + (" " if i < len(words) - 1 else ""),
            "index": i,
            "session_id": session_id,
        }
        await asyncio.sleep(0.04)

    # Final message
    yield {
        "type": "done",
        "session_id": session_id,
        "message_count": len(words),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 1 — WebSocket Auth
# ─────────────────────────────────────────────────────────────────────────────
# HTTP headers aren't easily accessible in WebSocket handshake from browsers.
# Common patterns:
#   1. Pass token as query param: ?token=eyJ...  (simplest)
#   2. Use a one-time ticket system (more secure)
#   3. Send token as first message after connect

def authenticate_ws_token(token: str | None) -> str | None:
    """
    Validates a WebSocket auth token.
    In production: call verify_access_token() from Module 04.
    Returns user_id if valid, None if invalid.
    """
    if not token:
        return None
    # Fake validation — in production this decodes a JWT
    if token.startswith("valid-"):
        return token.replace("valid-", "user-")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("✅ WebSocket server started")
    yield
    print("✅ WebSocket server stopped")


app = FastAPI(
    title="Module 09 — WebSockets & Real-Time",
    description="WebSocket agent streaming, connection manager, session persistence",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 1 — Basic WebSocket Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/echo")
async def ws_echo(websocket: WebSocket, token: str | None = Query(None)) -> None:
    """
    LESSON 1 — Simplest WebSocket: echo server with auth.

    Connect with: ws://localhost:8000/ws/echo?token=valid-abc123
    Test in the browser demo at /ws-demo

    WebSocket lifecycle:
      1. Client sends HTTP upgrade request
      2. Server calls websocket.accept() → connection open
      3. Both sides can send/receive messages
      4. Either side closes → WebSocketDisconnect raised
    """
    user_id = authenticate_ws_token(token)
    if not user_id:
        # Reject connection before accepting — no 401 in WS, use close code
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()

    try:
        await websocket.send_json({
            "type": "connected",
            "user_id": user_id,
            "message": f"Hello {user_id}! Send me a message.",
        })

        while True:
            # Blocks here until a message arrives or the connection drops
            data = await websocket.receive_json()
            message_text = data.get("message", "")

            await websocket.send_json({
                "type": "echo",
                "original": message_text,
                "response": f"Echo: {message_text}",
                "timestamp": datetime.utcnow().isoformat(),
            })

    except WebSocketDisconnect as e:
        print(f"[WS] Client {user_id} disconnected: code={e.code}")


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 2 & 4 — Agent WebSocket with Session
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/agent/{session_id}")
async def ws_agent(
    websocket: WebSocket,
    session_id: str,
    token: str | None = Query(None),
) -> None:
    """
    LESSON 2 & 4 — Full agent WebSocket with:
    - Authentication
    - Session resumption (reconnect mid-conversation)
    - Streaming agent responses (thinking → tool_call → tokens → done)
    - Proper disconnect handling

    Connect: ws://localhost:8000/ws/agent/{session_id}?token=valid-user123
    Create a session first via POST /sessions, use the session_id here.
    """
    user_id = authenticate_ws_token(token)
    if not user_id:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    session = get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    if session.user_id != user_id:
        await websocket.close(code=4003, reason="Forbidden")
        return

    client_id = str(uuid.uuid4())
    await manager.connect(websocket, room_id=session_id, client_id=client_id, user_id=user_id)

    try:
        # Send connection confirmation + session history (for reconnection)
        await websocket.send_json({
            "type": "session_ready",
            "session_id": session_id,
            "message_history": session.messages[-10:],  # last 10 messages on reconnect
            "agent_id": session.agent_id,
        })

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "user_message":
                user_text = data.get("content", "")

                # Save user message to session
                user_msg = {
                    "role": "user",
                    "content": user_text,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                append_to_session(session_id, user_msg)

                # Stream agent response
                full_response = []
                async for chunk in stream_agent_response(user_text, session_id):
                    await websocket.send_json(chunk)
                    if chunk["type"] == "token":
                        full_response.append(chunk["delta"])

                # Save agent response to session
                append_to_session(session_id, {
                    "role": "assistant",
                    "content": "".join(full_response),
                    "timestamp": datetime.utcnow().isoformat(),
                })

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})

            elif msg_type == "pause_session":
                session.status = "paused"
                await websocket.send_json({"type": "session_paused", "session_id": session_id})

    except WebSocketDisconnect:
        manager.disconnect(room_id=session_id, client_id=client_id)
        print(f"[WS] Agent session {session_id} client {client_id} disconnected")


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 3 — Chat Room with Broadcasting
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/room/{room_id}")
async def ws_room(
    websocket: WebSocket,
    room_id: str,
    token: str | None = Query(None),
) -> None:
    """
    LESSON 3 — Multi-client room with broadcasting.
    When one client sends a message, ALL clients in the room receive it.
    This is the foundation for collaborative agent UIs.
    """
    user_id = authenticate_ws_token(token) or f"anon-{uuid.uuid4()!s:.8}"
    client_id = str(uuid.uuid4())

    await manager.connect(websocket, room_id=room_id, client_id=client_id, user_id=user_id)

    # Notify room that a new user joined
    await manager.broadcast_to_room(room_id, {
        "type": "user_joined",
        "user_id": user_id,
        "room_info": manager.room_info(room_id),
    })

    try:
        while True:
            data = await websocket.receive_json()

            # Broadcast message to entire room
            await manager.broadcast_to_room(room_id, {
                "type": "message",
                "from": user_id,
                "content": data.get("content", ""),
                "timestamp": datetime.utcnow().isoformat(),
            })

    except WebSocketDisconnect:
        manager.disconnect(room_id=room_id, client_id=client_id)
        await manager.broadcast_to_room(room_id, {
            "type": "user_left",
            "user_id": user_id,
            "room_info": manager.room_info(room_id),
        })


# ─────────────────────────────────────────────────────────────────────────────
# HTTP ROUTES — REST API companion to WebSocket endpoints
# ─────────────────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    user_id: str
    agent_id: str = "default-agent"


@app.post("/sessions", response_model=AgentSession, status_code=201, tags=["Sessions"])
async def create_session_endpoint(data: SessionCreate) -> AgentSession:
    """Create an agent session. Use the returned session_id in the WebSocket URL."""
    return create_session(user_id=data.user_id, agent_id=data.agent_id)


@app.get("/sessions/{session_id}", response_model=AgentSession, tags=["Sessions"])
async def get_session_endpoint(session_id: str) -> AgentSession:
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.get("/ws/stats", tags=["WebSocket"])
async def ws_stats() -> dict[str, Any]:
    return {
        "total_connections": manager.total_connections,
        "active_sessions": len(_sessions),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BROWSER TEST CLIENT
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/ws-demo", response_class=HTMLResponse, tags=["Demo"])
async def ws_demo() -> str:
    return """
    <!DOCTYPE html>
    <html>
    <head>
    <title>WebSocket Agent Demo</title>
    <style>
      body { font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }
      h2 { color: #58a6ff; }
      .controls { display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
      input, button, select { padding: 8px 12px; border-radius: 6px; border: 1px solid #30363d; background: #161b22; color: #c9d1d9; }
      button { cursor: pointer; background: #1f6feb; border-color: #1f6feb; color: white; }
      button:hover { background: #388bfd; }
      #log { background: #161b22; border: 1px solid #30363d; padding: 16px; border-radius: 8px; height: 400px; overflow-y: auto; white-space: pre-wrap; font-size: 12px; }
      .msg-user { color: #79c0ff; }
      .msg-agent { color: #56d364; }
      .msg-tool { color: #e3b341; }
      .msg-system { color: #8b949e; }
    </style>
    </head>
    <body>
    <h2>WebSocket Agent Demo — Module 09</h2>

    <div class="controls">
      <input id="token" value="valid-user123" placeholder="Token (use valid-...)" style="width:200px">
      <button onclick="createSession()">1. Create Session</button>
      <button onclick="connectAgent()" id="connectBtn">2. Connect WebSocket</button>
      <button onclick="disconnect()" id="disconnectBtn" disabled>Disconnect</button>
    </div>

    <div class="controls">
      <input id="message" value="Search for Python books" placeholder="Your message" style="width:300px">
      <button onclick="sendMessage()" id="sendBtn" disabled>Send Message</button>
    </div>

    <div id="sessionInfo" style="margin-bottom:10px;color:#8b949e;font-size:12px">No session yet</div>
    <div id="log"></div>

    <script>
    let ws = null;
    let sessionId = null;

    function log(msg, cls = 'msg-system') {
      const el = document.getElementById('log');
      const line = document.createElement('div');
      line.className = cls;
      line.textContent = '[' + new Date().toLocaleTimeString() + '] ' + (typeof msg === 'object' ? JSON.stringify(msg, null, 2) : msg);
      el.appendChild(line);
      el.scrollTop = el.scrollHeight;
    }

    async function createSession() {
      const token = document.getElementById('token').value;
      const userId = token.replace('valid-', 'user-');
      const resp = await fetch('/sessions', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({user_id: userId, agent_id: 'demo-agent'})
      });
      const data = await resp.json();
      sessionId = data.session_id;
      document.getElementById('sessionInfo').textContent = 'Session: ' + sessionId;
      log('Session created: ' + sessionId);
    }

    function connectAgent() {
      if (!sessionId) { alert('Create a session first!'); return; }
      const token = document.getElementById('token').value;
      ws = new WebSocket('ws://localhost:8000/ws/agent/' + sessionId + '?token=' + token);

      ws.onopen = () => {
        log('WebSocket connected ✓', 'msg-agent');
        document.getElementById('connectBtn').disabled = true;
        document.getElementById('disconnectBtn').disabled = false;
        document.getElementById('sendBtn').disabled = false;
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'token') {
          process.stdout ? null : null; // just render
          const logEl = document.getElementById('log');
          const lastDiv = logEl.lastChild;
          if (lastDiv && lastDiv.className === 'msg-agent' && lastDiv.dataset.streaming) {
            lastDiv.textContent += data.delta;
          } else {
            const div = document.createElement('div');
            div.className = 'msg-agent';
            div.dataset.streaming = true;
            div.textContent = '[token] ' + data.delta;
            logEl.appendChild(div);
          }
          logEl.scrollTop = logEl.scrollHeight;
        } else if (data.type === 'tool_call') {
          log('🔧 Tool call: ' + data.tool_name + '(' + JSON.stringify(data.tool_input) + ')', 'msg-tool');
        } else if (data.type === 'tool_result') {
          log('📦 Tool result: ' + JSON.stringify(data.result), 'msg-tool');
        } else if (data.type === 'thinking') {
          log('💭 ' + data.content, 'msg-system');
        } else if (data.type === 'done') {
          log('✓ Response complete', 'msg-agent');
        } else {
          log(data, 'msg-system');
        }
      };

      ws.onclose = (e) => {
        log('Disconnected: code=' + e.code + ' reason=' + e.reason, 'msg-system');
        document.getElementById('connectBtn').disabled = false;
        document.getElementById('disconnectBtn').disabled = true;
        document.getElementById('sendBtn').disabled = true;
      };

      ws.onerror = (e) => log('Error: ' + e.message, 'msg-tool');
    }

    function sendMessage() {
      const msg = document.getElementById('message').value;
      if (!msg || !ws) return;
      ws.send(JSON.stringify({type: 'user_message', content: msg}));
      log('You: ' + msg, 'msg-user');
    }

    function disconnect() {
      if (ws) ws.close();
    }

    document.getElementById('message').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') sendMessage();
    });
    </script>
    </body>
    </html>
    """


# ─────────────────────────────────────────────────────────────────────────────
# PRACTICE EXERCISES
# ─────────────────────────────────────────────────────────────────────────────
# 1. Add message replay: when a client reconnects, send the last 50 messages.
# 2. Implement heartbeat: server sends ping every 30s, disconnects idle clients.
# 3. Add typing indicators: "agent is typing..." sent before the response.
# 4. Implement rooms with password: clients must send a room password to join.
# 5. Add Redis pub/sub: broadcast room messages across multiple server instances.
