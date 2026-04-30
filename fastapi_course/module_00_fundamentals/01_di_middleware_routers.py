"""
MODULE 0 — FastAPI Fundamentals (Part 2)
=========================================
Topics:
  0.11  Dependency Injection basics (Depends)
  0.12  Middleware basics
  0.13  APIRouter — splitting code into multiple files
  0.14  Background tasks
  0.15  OpenAPI customization
  0.16  CORS basics
  0.17  Static files
  0.18  Testing basics with TestClient
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.11  DEPENDENCY INJECTION — Depends()
# ─────────────────────────────────────────────────────────────────────────────
"""
Dependencies are functions that FastAPI calls BEFORE your endpoint function.
The result is injected as a parameter.

Why use them:
  - Avoid repeating the same logic in every endpoint (auth check, DB session, etc.)
  - Share state between functions
  - Easily swappable for testing (override dependencies)
"""

from fastapi import FastAPI, Depends, HTTPException, Header, Query, status
from pydantic import BaseModel

app = FastAPI(title="Module 0 Part 2")


# Simple dependency — a function
def get_api_version() -> str:
    return "1.0.0"

@app.get("/version")
async def get_version(version: str = Depends(get_api_version)):
    return {"api_version": version}


# Dependency with parameters — validates a common query param
async def pagination(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
) -> dict:
    return {"skip": skip, "limit": limit}

@app.get("/posts")
async def list_posts(pages: dict = Depends(pagination)):
    return {"skip": pages["skip"], "limit": pages["limit"], "posts": []}

@app.get("/comments")
async def list_comments(pages: dict = Depends(pagination)):
    # Same pagination logic reused — no copy-paste
    return {"skip": pages["skip"], "limit": pages["limit"], "comments": []}


# Dependency that raises HTTPException (auth guard)
async def verify_token(x_token: str | None = Header(None)) -> str:
    if not x_token or x_token != "valid-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Token header missing or invalid",
        )
    return x_token

@app.get("/secure-data")
async def secure_data(token: str = Depends(verify_token)):
    return {"data": "secret", "verified_with_token": token}


# Class-based dependency — useful for grouping related params
class CommonQueryParams:
    def __init__(
        self,
        q: str | None = None,
        skip: int = 0,
        limit: int = 10,
        sort_by: str = "created_at",
    ):
        self.q = q
        self.skip = skip
        self.limit = limit
        self.sort_by = sort_by

@app.get("/items")
async def list_items(commons: CommonQueryParams = Depends()):
    # Depends() with no args on a class → auto-instantiates it
    return {
        "search": commons.q,
        "skip": commons.skip,
        "limit": commons.limit,
        "sort_by": commons.sort_by,
    }


# Chained dependencies — a dependency can itself depend on others
async def get_db_connection():
    """Simulates getting a DB connection."""
    db = {"connection": "fake-db-conn"}
    try:
        yield db       # yield makes this a generator dependency
    finally:
        pass           # cleanup here (close connection, etc.)

async def get_current_user(
    token: str = Depends(verify_token),     # depends on token check
    db: dict = Depends(get_db_connection),  # depends on DB
) -> dict:
    # In real app: query DB with token to get user
    return {"id": 1, "email": "user@example.com", "role": "admin"}

@app.get("/profile")
async def get_profile(user: dict = Depends(get_current_user)):
    return {"profile": user}


# Dependencies at router/app level — applied to ALL routes under it
from fastapi import APIRouter

admin_router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(verify_token)],  # ALL admin routes need valid token
)

@admin_router.get("/dashboard")
async def admin_dashboard():
    return {"admin": True, "stats": {}}

@admin_router.delete("/users/{user_id}")
async def admin_delete_user(user_id: int):
    return {"deleted": user_id}

app.include_router(admin_router)


# ─────────────────────────────────────────────────────────────────────────────
# 0.12  MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────
"""
Middleware runs on EVERY request before it reaches your endpoint,
and on EVERY response before it's sent back.

Use for: logging, request timing, auth, rate limiting, adding headers.
"""

import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)   # call the actual endpoint
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id    # attach to request state
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        print(f"→ {request.method} {request.url.path}")
        response = await call_next(request)
        print(f"← {response.status_code}")
        return response


app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(LoggingMiddleware)
# Note: middleware is applied in reverse order of registration


# ─────────────────────────────────────────────────────────────────────────────
# 0.13  APIRouter — splitting into multiple files
# ─────────────────────────────────────────────────────────────────────────────
"""
In a real project, you NEVER define all routes in main.py.
Each feature gets its own file with its own APIRouter.

File: routers/products.py
"""

products_router = APIRouter(prefix="/products", tags=["products"])

PRODUCTS = {
    1: {"name": "Laptop", "price": 999},
    2: {"name": "Phone", "price": 499},
}

@products_router.get("/")
async def list_products():
    return list(PRODUCTS.values())

@products_router.get("/{product_id}")
async def get_product(product_id: int):
    if product_id not in PRODUCTS:
        raise HTTPException(status_code=404, detail="Product not found")
    return PRODUCTS[product_id]

@products_router.post("/", status_code=201)
async def create_product(name: str, price: float):
    new_id = max(PRODUCTS.keys()) + 1
    PRODUCTS[new_id] = {"name": name, "price": price}
    return {"id": new_id, **PRODUCTS[new_id]}

@products_router.put("/{product_id}")
async def update_product(product_id: int, name: str, price: float):
    if product_id not in PRODUCTS:
        raise HTTPException(status_code=404, detail="Product not found")
    PRODUCTS[product_id] = {"name": name, "price": price}
    return PRODUCTS[product_id]

@products_router.delete("/{product_id}", status_code=204)
async def delete_product(product_id: int):
    if product_id not in PRODUCTS:
        raise HTTPException(status_code=404, detail="Product not found")
    del PRODUCTS[product_id]
    # 204 No Content — return nothing


# In main.py you would do:
# from routers.products import products_router
# app.include_router(products_router)
app.include_router(products_router)


# ─────────────────────────────────────────────────────────────────────────────
# 0.14  BACKGROUND TASKS
# ─────────────────────────────────────────────────────────────────────────────
"""
BackgroundTasks lets you run work AFTER the response is sent.
The client doesn't wait for it.
Use for: sending emails, logging, notifications, webhooks.
"""

from fastapi import BackgroundTasks

def send_welcome_email(email: str, username: str):
    """This runs after the response is already sent to client."""
    import time
    time.sleep(2)    # simulate slow email sending
    print(f"Email sent to {email}: Welcome, {username}!")


def log_signup(username: str, ip: str):
    print(f"New signup: {username} from {ip}")


class UserSignup(BaseModel):
    username: str
    email: str

@app.post("/signup")
async def signup(
    user: UserSignup,
    background_tasks: BackgroundTasks,
    request: Request,
):
    # Response is sent immediately — background tasks run after
    background_tasks.add_task(send_welcome_email, user.email, user.username)
    background_tasks.add_task(log_signup, user.username, request.client.host)

    return {"message": f"Welcome {user.username}! Check your email."}
    # At this point, send_welcome_email starts running in background


# ─────────────────────────────────────────────────────────────────────────────
# 0.15  OPENAPI CUSTOMIZATION
# ─────────────────────────────────────────────────────────────────────────────
"""
FastAPI generates /openapi.json automatically.
You can customize the docs appearance and add extra info.
"""

# In main.py, customize FastAPI() constructor:
custom_app = FastAPI(
    title="My Production API",
    description="""
## My API

This API does amazing things.

### Features
- User management
- Product catalog
- Payment processing
    """,
    version="2.0.0",
    terms_of_service="https://example.com/terms",
    contact={
        "name": "Qasim Khan",
        "email": "qasimmkhan91@gmail.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {"name": "users", "description": "User management operations"},
        {"name": "products", "description": "Product catalog"},
        {"name": "admin", "description": "Admin-only operations"},
    ],
    docs_url="/docs",          # Swagger UI URL
    redoc_url="/redoc",        # ReDoc URL
    openapi_url="/openapi.json",
)


# ─────────────────────────────────────────────────────────────────────────────
# 0.16  CORS
# ─────────────────────────────────────────────────────────────────────────────
"""
CORS (Cross-Origin Resource Sharing) — required when your frontend (React, Vue)
is on a different domain/port than your API.

Without CORS: browser blocks API calls from frontend.
"""

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",       # React dev server
        "http://localhost:5173",       # Vite dev server
        "https://yourdomain.com",      # Production frontend
    ],
    allow_credentials=True,            # allow cookies
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],               # allow all headers
    expose_headers=["X-Request-ID"],   # expose custom headers to browser
)

# For development only — allow ALL origins (never use in production):
# allow_origins=["*"]


# ─────────────────────────────────────────────────────────────────────────────
# 0.17  STATIC FILES
# ─────────────────────────────────────────────────────────────────────────────
"""
Serve static files (images, CSS, JS) directly from FastAPI.
pip install aiofiles
"""

# from fastapi.staticfiles import StaticFiles
# app.mount("/static", StaticFiles(directory="static"), name="static")
# Now: http://localhost:8000/static/image.png → serves static/image.png


# ─────────────────────────────────────────────────────────────────────────────
# 0.18  TESTING WITH TestClient
# ─────────────────────────────────────────────────────────────────────────────
"""
FastAPI provides a synchronous TestClient (wraps httpx) for testing.
No async needed in tests — TestClient handles it.
pip install pytest httpx
"""

# tests/test_basics.py
"""
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello FastAPI"}


def test_get_item():
    response = client.get("/items/42")
    assert response.status_code == 200
    assert response.json() == {"item_id": 42}


def test_get_item_invalid():
    response = client.get("/items/abc")
    assert response.status_code == 422  # validation error


def test_create_item():
    response = client.post("/items", json={
        "name": "Test Item",
        "price": 9.99,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["item"]["name"] == "Test Item"


def test_missing_required_field():
    response = client.post("/items", json={"name": "No price"})
    assert response.status_code == 422


def test_protected_route_no_token():
    response = client.get("/secure-data")
    assert response.status_code == 401


def test_protected_route_with_token():
    response = client.get("/secure-data", headers={"X-Token": "valid-token"})
    assert response.status_code == 200


def test_override_dependency():
    # Replace a dependency for testing
    def mock_verify_token():
        return "test-token"

    app.dependency_overrides[verify_token] = mock_verify_token
    response = client.get("/secure-data")
    assert response.status_code == 200
    app.dependency_overrides.clear()
"""
