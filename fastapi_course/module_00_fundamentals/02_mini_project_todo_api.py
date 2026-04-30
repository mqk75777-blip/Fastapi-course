"""
MODULE 0 — Mini Project: Complete Todo API
==========================================
Ties all fundamentals together into one working project.

Covers:
  - CRUD endpoints (GET, POST, PUT, PATCH, DELETE)
  - Path + query + body params together
  - Pydantic models (create, update, response)
  - HTTPException for all error cases
  - Dependency injection (pagination, fake auth)
  - Background tasks (notification simulation)
  - APIRouter with prefix and tags
  - In-memory "database" (replaced with real DB in Module 5)

Run:
    pip install fastapi uvicorn
    uvicorn 02_mini_project:app --reload
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Header,
    Query,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class TodoStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    done = "done"
    cancelled = "cancelled"


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class TodoCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str | None = Field(None, max_length=1000)
    priority: Priority = Priority.medium
    assignee_email: str | None = None
    due_date: str | None = None          # ISO date string


class TodoUpdate(BaseModel):
    """All optional for PATCH."""
    title: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = None
    priority: Priority | None = None
    status: TodoStatus | None = None
    assignee_email: str | None = None
    due_date: str | None = None


class TodoResponse(BaseModel):
    id: str
    title: str
    description: str | None
    priority: Priority
    status: TodoStatus
    assignee_email: str | None
    due_date: str | None
    created_at: str
    updated_at: str


class PaginatedTodos(BaseModel):
    items: list[TodoResponse]
    total: int
    skip: int
    limit: int


# ─── In-memory "database" ─────────────────────────────────────────────────────

DB: dict[str, dict] = {}   # todo_id → todo dict


def _now() -> str:
    return datetime.utcnow().isoformat()


# ─── Dependencies ─────────────────────────────────────────────────────────────

class PaginationParams:
    def __init__(
        self,
        skip: int = Query(0, ge=0, description="Records to skip"),
        limit: int = Query(10, ge=1, le=100, description="Max records to return"),
    ):
        self.skip = skip
        self.limit = limit


async def require_auth(x_api_key: str | None = Header(None)) -> str:
    """Simple API key gate. Module 4 replaces this with real JWT."""
    if not x_api_key or x_api_key != "my-secret-key":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Api-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


AuthDep = Annotated[str, Depends(require_auth)]
PagesDep = Annotated[PaginationParams, Depends()]


# ─── Background task ──────────────────────────────────────────────────────────

def notify_assignee(email: str, todo_title: str, action: str) -> None:
    """Simulates sending an email notification (runs in background)."""
    print(f"[EMAIL] → {email}: Todo '{todo_title}' was {action}")


# ─── Router ───────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/todos", tags=["todos"])


@router.get("/", response_model=PaginatedTodos)
async def list_todos(
    pages: PagesDep,
    _auth: AuthDep,
    status_filter: TodoStatus | None = Query(None, alias="status"),
    priority: Priority | None = None,
    q: str | None = Query(None, min_length=1, description="Search in title"),
):
    """List all todos with optional filtering and pagination."""
    items = list(DB.values())

    # Apply filters
    if status_filter:
        items = [i for i in items if i["status"] == status_filter]
    if priority:
        items = [i for i in items if i["priority"] == priority]
    if q:
        items = [i for i in items if q.lower() in i["title"].lower()]

    total = len(items)
    paginated = items[pages.skip : pages.skip + pages.limit]

    return PaginatedTodos(
        items=[TodoResponse(**i) for i in paginated],
        total=total,
        skip=pages.skip,
        limit=pages.limit,
    )


@router.post("/", response_model=TodoResponse, status_code=status.HTTP_201_CREATED)
async def create_todo(
    body: TodoCreate,
    background_tasks: BackgroundTasks,
    _auth: AuthDep,
):
    """Create a new todo. Notifies assignee in background if set."""
    todo_id = str(uuid.uuid4())
    now = _now()
    todo = {
        "id": todo_id,
        "title": body.title,
        "description": body.description,
        "priority": body.priority,
        "status": TodoStatus.pending,
        "assignee_email": body.assignee_email,
        "due_date": body.due_date,
        "created_at": now,
        "updated_at": now,
    }
    DB[todo_id] = todo

    if body.assignee_email:
        background_tasks.add_task(notify_assignee, body.assignee_email, body.title, "created")

    return TodoResponse(**todo)


@router.get("/{todo_id}", response_model=TodoResponse)
async def get_todo(todo_id: str, _auth: AuthDep):
    """Get a single todo by ID."""
    todo = DB.get(todo_id)
    if not todo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Todo '{todo_id}' not found",
        )
    return TodoResponse(**todo)


@router.patch("/{todo_id}", response_model=TodoResponse)
async def update_todo(
    todo_id: str,
    body: TodoUpdate,
    background_tasks: BackgroundTasks,
    _auth: AuthDep,
):
    """Partially update a todo. Only provided fields are changed."""
    todo = DB.get(todo_id)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")

    updates = body.model_dump(exclude_none=True)  # only fields the client sent
    todo.update(updates)
    todo["updated_at"] = _now()
    DB[todo_id] = todo

    if todo.get("assignee_email") and "status" in updates:
        background_tasks.add_task(
            notify_assignee,
            todo["assignee_email"],
            todo["title"],
            f"updated to {updates['status']}",
        )

    return TodoResponse(**todo)


@router.put("/{todo_id}", response_model=TodoResponse)
async def replace_todo(todo_id: str, body: TodoCreate, _auth: AuthDep):
    """Full replacement — PUT replaces the entire resource."""
    if todo_id not in DB:
        raise HTTPException(status_code=404, detail="Todo not found")

    existing = DB[todo_id]
    todo = {
        "id": todo_id,
        "status": existing["status"],       # preserve status on replace
        "created_at": existing["created_at"],
        "updated_at": _now(),
        **body.model_dump(),
    }
    DB[todo_id] = todo
    return TodoResponse(**todo)


@router.delete("/{todo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_todo(todo_id: str, _auth: AuthDep, background_tasks: BackgroundTasks):
    """Hard delete. Returns 204 No Content."""
    todo = DB.get(todo_id)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")

    if todo.get("assignee_email"):
        background_tasks.add_task(
            notify_assignee, todo["assignee_email"], todo["title"], "deleted"
        )

    del DB[todo_id]
    # 204 → return nothing (not even an empty dict)


@router.post("/{todo_id}/complete", response_model=TodoResponse)
async def complete_todo(
    todo_id: str,
    _auth: AuthDep,
    background_tasks: BackgroundTasks,
):
    """Convenience endpoint to mark a todo as done."""
    todo = DB.get(todo_id)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    if todo["status"] == TodoStatus.done:
        raise HTTPException(status_code=400, detail="Todo is already completed")
    if todo["status"] == TodoStatus.cancelled:
        raise HTTPException(status_code=400, detail="Cannot complete a cancelled todo")

    todo["status"] = TodoStatus.done
    todo["updated_at"] = _now()

    if todo.get("assignee_email"):
        background_tasks.add_task(notify_assignee, todo["assignee_email"], todo["title"], "completed")

    return TodoResponse(**todo)


@router.get("/stats/summary")
async def get_stats(_auth: AuthDep):
    """Aggregate statistics — no pagination needed."""
    todos = list(DB.values())
    by_status = {}
    by_priority = {}
    for t in todos:
        by_status[t["status"]] = by_status.get(t["status"], 0) + 1
        by_priority[t["priority"]] = by_priority.get(t["priority"], 0) + 1
    return {
        "total": len(todos),
        "by_status": by_status,
        "by_priority": by_priority,
        "completion_rate": round(
            by_status.get("done", 0) / len(todos) * 100, 1
        ) if todos else 0,
    }


# ─── App assembly ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Todo API — Module 0 Mini Project",
    description="A complete CRUD API demonstrating all FastAPI fundamentals",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # open for dev; lock down in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "todos_in_memory": len(DB)}


# ─── Seed data for testing ────────────────────────────────────────────────────

@app.on_event("startup")  # simple alternative to lifespan (for learning)
async def seed():
    for i in range(1, 4):
        tid = str(uuid.uuid4())
        DB[tid] = {
            "id": tid,
            "title": f"Sample Todo {i}",
            "description": f"This is sample todo number {i}",
            "priority": ["low", "medium", "high"][i % 3],
            "status": "pending",
            "assignee_email": f"user{i}@example.com",
            "due_date": None,
            "created_at": _now(),
            "updated_at": _now(),
        }


# ─── Quick test script (run manually) ────────────────────────────────────────
"""
pip install httpx

import httpx

BASE = "http://localhost:8000/api/v1"
HEADERS = {"X-Api-Key": "my-secret-key"}

# List todos
r = httpx.get(f"{BASE}/todos/", headers=HEADERS)
print(r.json())

# Create
r = httpx.post(f"{BASE}/todos/", headers=HEADERS, json={
    "title": "Build FastAPI project",
    "priority": "high",
    "assignee_email": "qasimmkhan91@gmail.com"
})
todo_id = r.json()["id"]

# Get
r = httpx.get(f"{BASE}/todos/{todo_id}", headers=HEADERS)
print(r.json())

# Patch
r = httpx.patch(f"{BASE}/todos/{todo_id}", headers=HEADERS, json={"status": "in_progress"})
print(r.json())

# Complete
r = httpx.post(f"{BASE}/todos/{todo_id}/complete", headers=HEADERS)
print(r.json())

# Stats
r = httpx.get(f"{BASE}/todos/stats/summary", headers=HEADERS)
print(r.json())

# Delete
r = httpx.delete(f"{BASE}/todos/{todo_id}", headers=HEADERS)
print(r.status_code)  # 204
"""
