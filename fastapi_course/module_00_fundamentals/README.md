# Module 0 — FastAPI Fundamentals

This module covers everything you need before touching production patterns.

## Files

| File | Topics |
|------|--------|
| `00_core_concepts.py` | Path params, query params, request body, response models, status codes, headers, cookies, forms, file uploads, HTTPException, response types |
| `01_di_middleware_routers.py` | Depends(), class-based deps, chained deps, middleware, APIRouter, background tasks, CORS, TestClient |
| `02_mini_project_todo_api.py` | Complete CRUD API tying everything together |

## Topics covered

### 00_core_concepts.py
- `0.1` First FastAPI app — how it works
- `0.2` Path parameters — `{item_id}`, type validation, Enum
- `0.3` Query parameters — required, optional, `Query()` with validation, list params
- `0.4` Request body — POST/PUT/PATCH with Pydantic, combining path + query + body
- `0.5` Response models — `response_model=`, `status_code=`, `response_model_exclude`
- `0.6` Headers — `Header()`, cookies `Cookie()`, setting response headers
- `0.7` Form data + file uploads — `Form()`, `UploadFile`, multiple files
- `0.8` HTTPException — 404, 400, 401, custom exception handlers
- `0.9` Path operation metadata — tags, summary, description, deprecated
- `0.10` Response types — `JSONResponse`, `HTMLResponse`, `RedirectResponse`, `StreamingResponse`

### 01_di_middleware_routers.py
- `0.11` Dependency injection — `Depends()`, generator deps with `yield`, chained deps, class-based deps, router-level deps
- `0.12` Middleware — `BaseHTTPMiddleware`, request timing, request ID, logging
- `0.13` APIRouter — splitting code into files, prefix, tags, router-level dependencies
- `0.14` Background tasks — `BackgroundTasks`, fire-and-forget after response
- `0.15` OpenAPI customization — title, description, contact, tags metadata
- `0.16` CORS — `CORSMiddleware`, allowed origins, credentials
- `0.17` Static files — serving files from disk
- `0.18` Testing — `TestClient`, `dependency_overrides`

### 02_mini_project_todo_api.py
Full Todo CRUD API with:
- `GET /todos/` — list with filtering (status, priority, search) + pagination
- `POST /todos/` — create with background email notification
- `GET /todos/{id}` — get single
- `PATCH /todos/{id}` — partial update (only sent fields change)
- `PUT /todos/{id}` — full replacement
- `DELETE /todos/{id}` — hard delete, 204 No Content
- `POST /todos/{id}/complete` — convenience action endpoint
- `GET /todos/stats/summary` — aggregation endpoint
- API key auth via dependency
- In-memory DB (dict) — replaced with real DB in Module 5

## Install & run

```bash
pip install fastapi uvicorn python-multipart

# Run the mini project
uvicorn 02_mini_project_todo_api:app --reload

# Docs: http://localhost:8000/docs
# Health: http://localhost:8000/health
```

## Key rules learned in this module

1. **Specific routes before parameterized**: `/users/me` must be before `/users/{id}`
2. **Never raise generic exceptions** in endpoints — always `HTTPException`
3. **response_model filters the output** — never manually exclude fields
4. **`exclude_none=True`** when applying PATCH updates — only change what was sent
5. **204 responses return nothing** — not even `{}`
6. **Dependencies can depend on dependencies** — chain them freely
7. **Background tasks run after the response is sent** — client doesn't wait
8. **CORS must be added before any route that the browser calls cross-origin**

## What Module 1 adds on top of this
Once you understand these fundamentals, Module 1 shows you how to organize
all of this into a production-grade folder structure with proper config,
lifespan management, and versioned API routing.
