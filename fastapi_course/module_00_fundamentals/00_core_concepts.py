"""
MODULE 0 — FastAPI Fundamentals
================================
Everything you need to understand BEFORE the production patterns.

Topics covered in this file:
  0.1  First FastAPI app
  0.2  Path parameters
  0.3  Query parameters
  0.4  Request body (POST/PUT/PATCH)
  0.5  Response models & status codes
  0.6  Headers & cookies
  0.7  Form data & file uploads
  0.8  HTTPException & error responses
  0.9  Path operations: tags, summary, description, deprecated
  0.10 Multiple HTTP methods on same path
"""

from fastapi import FastAPI

app = FastAPI(title="Module 0 — FastAPI Fundamentals", version="0.1.0")


# ─────────────────────────────────────────────────────────────────────────────
# 0.1  FIRST APP — how FastAPI works
# ─────────────────────────────────────────────────────────────────────────────
"""
FastAPI is built on top of Starlette (ASGI framework) and uses Pydantic for
data validation. When you define a route, FastAPI automatically:
  - Validates incoming data
  - Generates OpenAPI docs (/docs and /redoc)
  - Returns proper JSON responses

Run with:
    pip install fastapi uvicorn
    uvicorn module_00_fundamentals:app --reload
"""

@app.get("/")
async def root():
    """Simplest possible endpoint. Returns a dict → FastAPI converts to JSON."""
    return {"message": "Hello FastAPI"}


# ─────────────────────────────────────────────────────────────────────────────
# 0.2  PATH PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
"""
Path params are parts of the URL. Declared with {param_name} in the path.
FastAPI automatically validates the type — if you say int, it rejects strings.
"""

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    """
    item_id is extracted from the URL and validated as int.
    GET /items/42   → {"item_id": 42}
    GET /items/abc  → 422 Unprocessable Entity (automatic)
    """
    return {"item_id": item_id}

@app.get("/users/{user_id}/orders/{order_id}")
async def get_user_order(user_id: int, order_id: int):
    """Multiple path params in one route."""
    return {"user_id": user_id, "order_id": order_id}


# IMPORTANT: order matters — specific routes must come BEFORE parameterized ones
@app.get("/users/me")       # ← this must be defined BEFORE /users/{user_id}
async def get_me():
    return {"user": "current logged-in user"}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}


# Path params with Enum — only allow specific values
from enum import Enum

class ModelName(str, Enum):
    gpt4 = "gpt4"
    claude = "claude"
    groq = "groq"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    """
    Only accepts "gpt4", "claude", or "groq".
    Anything else → 422 automatically.
    Documented as enum in /docs.
    """
    if model_name == ModelName.groq:
        return {"model": model_name, "fastest": True}
    return {"model": model_name}


# ─────────────────────────────────────────────────────────────────────────────
# 0.3  QUERY PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
"""
Any function parameter that is NOT in the path is treated as a query param.
GET /search?q=fastapi&limit=10&skip=0
"""

@app.get("/search")
async def search(
    q: str,                    # required — no default
    limit: int = 10,           # optional, default 10
    skip: int = 0,             # optional, default 0
    active: bool = True,       # bools: ?active=true/false/1/0 all work
):
    return {"query": q, "limit": limit, "skip": skip, "active": active}


# Optional query params — use | None with a default of None
@app.get("/products")
async def list_products(
    category: str | None = None,     # optional, not required
    min_price: float | None = None,
    max_price: float | None = None,
):
    filters = {}
    if category:
        filters["category"] = category
    if min_price is not None:
        filters["min_price"] = min_price
    if max_price is not None:
        filters["max_price"] = max_price
    return {"filters": filters, "results": []}


# Query params with validation using Query()
from fastapi import Query

@app.get("/validated-search")
async def validated_search(
    q: str = Query(min_length=3, max_length=50, description="Search term"),
    limit: int = Query(default=10, ge=1, le=100, description="Results per page"),
    tags: list[str] = Query(default=[], description="Filter by tags"),
    # ?tags=python&tags=fastapi → tags=["python", "fastapi"]
):
    return {"q": q, "limit": limit, "tags": tags}


# ─────────────────────────────────────────────────────────────────────────────
# 0.4  REQUEST BODY — POST / PUT / PATCH
# ─────────────────────────────────────────────────────────────────────────────
"""
For POST/PUT/PATCH, data comes in the request body as JSON.
Define a Pydantic model — FastAPI validates and parses it automatically.
"""

from pydantic import BaseModel, Field, EmailStr

class ItemCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0, description="Must be positive")
    description: str | None = None
    in_stock: bool = True


@app.post("/items")
async def create_item(item: ItemCreate):
    """
    Client sends JSON body:
    {
        "name": "Laptop",
        "price": 999.99,
        "description": "Gaming laptop"
    }
    FastAPI validates it against ItemCreate before your function runs.
    """
    return {"created": True, "item": item.model_dump()}


# Combining path param + query param + body in one endpoint
@app.put("/items/{item_id}")
async def update_item(
    item_id: int,               # from path
    notify: bool = False,       # from query string
    item: ItemCreate = ...,     # from request body (... = required)
):
    return {
        "item_id": item_id,
        "notify_on_update": notify,
        "updated_data": item.model_dump(),
    }


# PATCH — partial update (all fields optional)
class ItemUpdate(BaseModel):
    name: str | None = None
    price: float | None = Field(None, gt=0)
    description: str | None = None
    in_stock: bool | None = None

@app.patch("/items/{item_id}")
async def partial_update(item_id: int, item: ItemUpdate):
    """Only update fields that are provided."""
    updates = item.model_dump(exclude_none=True)  # drops None fields
    return {"item_id": item_id, "applied_updates": updates}


# Multiple body params — use Body()
from fastapi import Body

class UserSchema(BaseModel):
    name: str
    email: EmailStr

class AddressSchema(BaseModel):
    street: str
    city: str

@app.post("/register")
async def register(
    user: UserSchema,
    address: AddressSchema,
    # Client must send: {"user": {...}, "address": {...}}
):
    return {"user": user.model_dump(), "address": address.model_dump()}


# ─────────────────────────────────────────────────────────────────────────────
# 0.5  RESPONSE MODELS & STATUS CODES
# ─────────────────────────────────────────────────────────────────────────────
"""
response_model= tells FastAPI:
  - What to include in the response (filters out extra fields)
  - What to document in /docs
  - Validate the response before sending (catches bugs)

status_code= sets the HTTP status code.
"""

from fastapi import status

class ItemResponse(BaseModel):
    id: int
    name: str
    price: float
    # notice: no 'description' — it will be filtered out even if present

@app.post(
    "/items/create",
    response_model=ItemResponse,
    status_code=status.HTTP_201_CREATED,  # 201 Created instead of 200
)
async def create_item_v2(item: ItemCreate):
    db_item = {"id": 1, "name": item.name, "price": item.price, "description": item.description}
    return db_item  # FastAPI filters to only id, name, price via response_model


# response_model_exclude — exclude specific fields from response
class UserWithPassword(BaseModel):
    id: int
    email: str
    hashed_password: str   # we don't want this in the response

@app.get(
    "/users/{user_id}/full",
    response_model=UserWithPassword,
    response_model_exclude={"hashed_password"},  # strip it from response
)
async def get_user_full(user_id: int):
    return {"id": user_id, "email": "user@example.com", "hashed_password": "secret"}


# Multiple response types — Union
from typing import Union

class SuccessResponse(BaseModel):
    success: bool = True
    data: dict

class NotFoundResponse(BaseModel):
    success: bool = False
    detail: str

@app.get(
    "/flexible/{item_id}",
    responses={
        200: {"model": SuccessResponse},
        404: {"model": NotFoundResponse},
    }
)
async def flexible_endpoint(item_id: int):
    if item_id == 0:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=404,
            content={"success": False, "detail": "Item not found"}
        )
    return {"success": True, "data": {"id": item_id}}


# ─────────────────────────────────────────────────────────────────────────────
# 0.6  HEADERS & COOKIES
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import Header, Cookie
from fastapi.responses import JSONResponse, Response

@app.get("/with-headers")
async def read_headers(
    user_agent: str | None = Header(None),          # reads User-Agent header
    x_api_version: str | None = Header(None),       # reads X-Api-Version header
    accept_language: str | None = Header(None),
):
    """
    FastAPI converts header names: x_api_version → X-Api-Version automatically.
    Headers are always strings.
    """
    return {
        "user_agent": user_agent,
        "api_version": x_api_version,
        "language": accept_language,
    }


@app.get("/set-cookie")
async def set_cookie(response: Response):
    """Set a cookie in the response."""
    response.set_cookie(
        key="session_id",
        value="abc123",
        httponly=True,       # not accessible via JS
        secure=True,         # HTTPS only
        samesite="lax",
        max_age=3600,        # 1 hour
    )
    return {"message": "Cookie set"}


@app.get("/read-cookie")
async def read_cookie(session_id: str | None = Cookie(None)):
    """Read a cookie from the request."""
    if not session_id:
        return {"logged_in": False}
    return {"logged_in": True, "session_id": session_id}


# Custom response headers
@app.get("/custom-headers")
async def custom_headers():
    content = {"message": "Response with custom headers"}
    headers = {
        "X-Custom-Header": "my-value",
        "X-Process-Time": "12ms",
    }
    return JSONResponse(content=content, headers=headers)


# ─────────────────────────────────────────────────────────────────────────────
# 0.7  FORM DATA & FILE UPLOADS
# ─────────────────────────────────────────────────────────────────────────────
"""
pip install python-multipart  ← required for forms and file uploads
"""

from fastapi import Form, File, UploadFile

@app.post("/login-form")
async def login_form(
    username: str = Form(...),    # from HTML form, not JSON body
    password: str = Form(...),
):
    """Accepts application/x-www-form-urlencoded (HTML form submission)."""
    return {"username": username, "authenticated": True}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts multipart/form-data.
    UploadFile gives you:
      - file.filename
      - file.content_type
      - await file.read()
      - await file.seek(0)
    """
    content = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content),
    }


@app.post("/upload-multiple")
async def upload_multiple(files: list[UploadFile] = File(...)):
    """Upload multiple files at once."""
    results = []
    for file in files:
        content = await file.read()
        results.append({"filename": file.filename, "size": len(content)})
    return {"uploaded": results}


@app.post("/upload-with-metadata")
async def upload_with_metadata(
    file: UploadFile = File(...),
    description: str = Form(""),     # mix file + form fields
    category: str = Form(...),
):
    content = await file.read()
    return {
        "filename": file.filename,
        "description": description,
        "category": category,
        "size": len(content),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 0.8  HTTPException & ERROR RESPONSES
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import HTTPException

FAKE_DB = {1: "Laptop", 2: "Phone", 3: "Tablet"}

@app.get("/products/{product_id}")
async def get_product(product_id: int):
    """
    HTTPException stops execution and returns an error response immediately.
    Never raise generic Python exceptions in endpoints — always HTTPException.
    """
    if product_id not in FAKE_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product {product_id} not found",
        )
    return {"id": product_id, "name": FAKE_DB[product_id]}


@app.delete("/products/{product_id}")
async def delete_product(product_id: int, confirm: bool = False):
    if product_id not in FAKE_DB:
        raise HTTPException(status_code=404, detail="Not found")
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pass ?confirm=true to confirm deletion",
        )
    return {"deleted": product_id}


# HTTPException with custom headers (useful for auth errors)
@app.get("/protected")
async def protected_route(token: str | None = Header(None)):
    if not token or token != "secret":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access": "granted"}


# Custom exception class + handler
from fastapi import Request
from fastapi.responses import JSONResponse

class InsufficientFundsError(Exception):
    def __init__(self, balance: float, required: float):
        self.balance = balance
        self.required = required

@app.exception_handler(InsufficientFundsError)
async def insufficient_funds_handler(request: Request, exc: InsufficientFundsError):
    return JSONResponse(
        status_code=402,
        content={
            "error": "insufficient_funds",
            "balance": exc.balance,
            "required": exc.required,
            "shortfall": exc.required - exc.balance,
        }
    )

@app.post("/pay")
async def make_payment(amount: float):
    balance = 50.0
    if amount > balance:
        raise InsufficientFundsError(balance=balance, required=amount)
    return {"paid": amount, "remaining": balance - amount}


# ─────────────────────────────────────────────────────────────────────────────
# 0.9  PATH OPERATION METADATA
# ─────────────────────────────────────────────────────────────────────────────
"""
All this metadata appears in /docs (Swagger UI) and /redoc.
In production APIs this is how you communicate contracts to frontend devs.
"""

@app.get(
    "/documented-endpoint",
    tags=["examples"],                         # groups in /docs sidebar
    summary="Short one-liner description",
    description="""
## Detailed description (supports Markdown)

This endpoint does something complex.

- Step 1: validates input
- Step 2: processes data
- Step 3: returns result
    """,
    response_description="The processed result",
    deprecated=False,                          # set True to show as deprecated in docs
)
async def documented_endpoint():
    return {"result": "documented"}


# Docstring as description — FastAPI reads it automatically
@app.get("/auto-documented", tags=["examples"])
async def auto_documented():
    """
    FastAPI uses this docstring as the endpoint description.
    No need to pass description= separately if docstring is enough.
    Returns a simple example response.
    """
    return {"example": True}


# ─────────────────────────────────────────────────────────────────────────────
# 0.10  RESPONSE TYPES — not always JSON
# ─────────────────────────────────────────────────────────────────────────────

from fastapi.responses import (
    JSONResponse,
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    FileResponse,
    StreamingResponse,
)

@app.get("/html", response_class=HTMLResponse)
async def return_html():
    return """
    <html>
        <body>
            <h1>Hello from FastAPI</h1>
            <p>This is HTML, not JSON.</p>
        </body>
    </html>
    """

@app.get("/plain", response_class=PlainTextResponse)
async def return_plain():
    return "This is plain text, no JSON encoding"

@app.get("/redirect")
async def redirect():
    return RedirectResponse(url="/", status_code=302)

@app.get("/download")
async def download_file():
    # FileResponse for serving actual files
    # return FileResponse("path/to/file.pdf", filename="report.pdf")
    return {"note": "would return a file in real usage"}

@app.get("/stream-bytes")
async def stream_bytes():
    """Streaming response — useful for large files, SSE, LLM tokens."""
    async def generate():
        for i in range(5):
            yield f"chunk {i}\n".encode()
    return StreamingResponse(generate(), media_type="text/plain")
