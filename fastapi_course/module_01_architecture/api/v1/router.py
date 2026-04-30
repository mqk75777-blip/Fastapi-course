"""
api/v1/router.py — root API router that aggregates all sub-routers
Add new feature routers here as the project grows.
"""

from fastapi import APIRouter
from api.v1.endpoints import users, items, agents

api_router = APIRouter()

api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(items.router, prefix="/items", tags=["items"])
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])


# ─── api/v1/endpoints/users.py ───────────────────────────────────────────────
"""
Example endpoint file. Each feature gets its own file.
"""

from fastapi import APIRouter, Depends, HTTPException, status

router = APIRouter()


@router.get("/me")
async def get_current_user():
    # Will wire to real auth in Module 4
    return {"id": 1, "email": "user@example.com", "role": "admin"}


@router.get("/{user_id}")
async def get_user(user_id: int):
    if user_id != 1:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return {"id": user_id, "email": "user@example.com"}
