"""
================================================================================
MODULE 04 — Authentication & Authorization
================================================================================
Topics:
  L1. OAuth2 password flow & JWT from scratch
  L2. Refresh token rotation with Redis (simulated with dict here)
  L3. RBAC — roles, permissions, scopes
  L4. API key auth for agent-to-API calls
  L5. Multi-tenant auth architecture

Run:  uvicorn main:app --reload
Docs: http://localhost:8000/docs
      Click "Authorize" → use admin@example.com / admin123
================================================================================
"""

from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import (
    APIKeyHeader,
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
    SecurityScopes,
)
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# In production: load from environment variables via pydantic-settings
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION_" + secrets.token_hex(16))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 3 — Roles & Permissions
# ─────────────────────────────────────────────────────────────────────────────

class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    AGENT = "agent"      # for AI agents calling the API
    READONLY = "readonly"


# Permission scopes — what each role can do
ROLE_SCOPES: dict[Role, list[str]] = {
    Role.ADMIN:    ["users:read", "users:write", "users:delete", "agents:manage", "admin:all"],
    Role.USER:     ["users:read", "agents:use"],
    Role.AGENT:    ["agents:use", "tools:call", "memory:read", "memory:write"],
    Role.READONLY: ["users:read"],
}


# ─────────────────────────────────────────────────────────────────────────────
# PASSWORD HASHING
# ─────────────────────────────────────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ─────────────────────────────────────────────────────────────────────────────
# FAKE DATABASE (replace with SQLAlchemy in real app)
# ─────────────────────────────────────────────────────────────────────────────

class StoredUser(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    email: str
    username: str
    hashed_password: str
    role: Role = Role.USER
    tenant_id: str = "default"   # LESSON 5: multi-tenancy
    is_active: bool = True


# Seed users
_users_db: dict[str, StoredUser] = {
    "admin@example.com": StoredUser(
        email="admin@example.com",
        username="admin",
        hashed_password=hash_password("admin123"),
        role=Role.ADMIN,
        tenant_id="tenant_acme",
    ),
    "user@example.com": StoredUser(
        email="user@example.com",
        username="regular_user",
        hashed_password=hash_password("user1234"),
        role=Role.USER,
        tenant_id="tenant_acme",
    ),
    "agent@example.com": StoredUser(
        email="agent@example.com",
        username="ai_agent",
        hashed_password=hash_password("agent123"),
        role=Role.AGENT,
        tenant_id="tenant_acme",
    ),
}

# LESSON 2: Refresh token store (use Redis in production)
# Structure: {token_hash: {user_email, expires_at, jti}}
_refresh_tokens: dict[str, dict[str, Any]] = {}

# LESSON 4: API key store
# Structure: {key_hash: {user_id, scopes, name, created_at}}
_api_keys: dict[str, dict[str, Any]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 1 — JWT Token Creation & Verification
# ─────────────────────────────────────────────────────────────────────────────

class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


class TokenData(BaseModel):
    sub: str          # subject (email)
    role: Role
    scopes: list[str]
    tenant_id: str
    jti: str          # JWT ID — unique per token, used to blacklist


def create_access_token(user: StoredUser) -> str:
    """
    Creates a signed JWT access token.
    Payload contains: subject, role, scopes, tenant, expiry, unique ID.
    """
    now = datetime.now(timezone.utc)
    scopes = ROLE_SCOPES.get(user.role, [])
    payload = {
        "sub": user.email,
        "role": user.role.value,
        "scopes": scopes,
        "tenant_id": user.tenant_id,
        "jti": str(uuid4()),         # unique token ID for blacklisting
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user: StoredUser) -> str:
    """
    Refresh tokens are opaque random strings stored server-side.
    Unlike JWTs, they can be individually revoked.
    """
    token = secrets.token_urlsafe(64)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    _refresh_tokens[token_hash] = {
        "user_email": user.email,
        "expires_at": expires_at,
        "jti": str(uuid4()),
    }
    return token


def verify_access_token(token: str) -> TokenData:
    """Decodes and validates a JWT. Raises 401 on any error."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return TokenData(
            sub=payload["sub"],
            role=Role(payload["role"]),
            scopes=payload.get("scopes", []),
            tenant_id=payload.get("tenant_id", "default"),
            jti=payload["jti"],
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI SECURITY SCHEMES
# ─────────────────────────────────────────────────────────────────────────────

# OAuth2PasswordBearer — reads Bearer token from Authorization header
# scopes dict documents available scopes in /docs UI
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/token",
    scopes={
        "users:read": "Read user data",
        "users:write": "Create and update users",
        "users:delete": "Delete users",
        "agents:manage": "Manage AI agents",
        "agents:use": "Use AI agent features",
        "tools:call": "Call registered tools",
        "memory:read": "Read agent memory",
        "memory:write": "Write agent memory",
        "admin:all": "Full admin access",
    },
)

# API Key — reads from X-API-Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 3 — Scope-Based Dependencies
# ─────────────────────────────────────────────────────────────────────────────

async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme),
) -> StoredUser:
    """
    The core auth dependency. Validates the JWT and checks required scopes.

    SecurityScopes is injected by FastAPI when you use Security() instead of Depends().
    It contains the scopes required by the current route.

    Usage in routes:
        user = Security(get_current_user, scopes=["users:read"])
    """
    authenticate_value = (
        f'Bearer scope="{security_scopes.scope_str}"'
        if security_scopes.scopes
        else "Bearer"
    )

    token_data = verify_access_token(token)

    user = _users_db.get(token_data.sub)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": authenticate_value},
        )

    # Check that token has all required scopes
    for required_scope in security_scopes.scopes:
        if required_scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required scope: {required_scope}",
                headers={"WWW-Authenticate": authenticate_value},
            )

    return user


def require_role(*roles: Role):
    """
    Dependency factory for role-based access control.
    Use: Depends(require_role(Role.ADMIN, Role.USER))
    """
    async def _check(user: StoredUser = Security(get_current_user)) -> StoredUser:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {user.role.value!r} not allowed. Required: {[r.value for r in roles]}",
            )
        return user
    return _check


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 4 — API Key Auth for Agents
# ─────────────────────────────────────────────────────────────────────────────

async def get_api_key_user(api_key: str | None = Depends(api_key_header)) -> StoredUser | None:
    """
    Validates an API key from the X-API-Key header.
    Returns the associated user or raises 401.
    """
    if not api_key:
        return None

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    key_data = _api_keys.get(key_hash)

    if not key_data:
        raise HTTPException(status_code=401, detail="Invalid API key")

    user = _users_db.get(key_data["user_email"])
    if not user:
        raise HTTPException(status_code=401, detail="API key owner not found")

    return user


async def get_current_user_flexible(
    jwt_user: StoredUser | None = Security(get_current_user, scopes=[]),
    api_key_user: StoredUser | None = Depends(get_api_key_user),
) -> StoredUser:
    """Accepts EITHER Bearer JWT OR X-API-Key. Useful for agent routes."""
    user = jwt_user or api_key_user
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 5 — Tenant-Aware Dependency
# ─────────────────────────────────────────────────────────────────────────────

class TenantContext(BaseModel):
    tenant_id: str
    user: StoredUser


async def get_tenant_context(
    user: StoredUser = Security(get_current_user, scopes=["users:read"]),
) -> TenantContext:
    """
    Extracts tenant context from the authenticated user.
    In a real app, you'd also fetch the tenant's config from DB here.
    All downstream operations use tenant_id to scope their queries.
    """
    return TenantContext(tenant_id=user.tenant_id, user=user)


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class UserPublic(BaseModel):
    id: str
    email: str
    username: str
    role: Role
    tenant_id: str


class RefreshRequest(BaseModel):
    refresh_token: str


class APIKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    scopes: list[str] = Field(default_factory=list)


class APIKeyResponse(BaseModel):
    key: str  # shown ONCE — never stored plain
    name: str
    scopes: list[str]
    created_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
# APP & ROUTES
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Module 04 — Auth & Authorization",
    description="JWT, RBAC, API keys, refresh token rotation, multi-tenancy",
    version="1.0.0",
)


# ── LESSON 1: Login endpoint ──────────────────────────────────────────────────

@app.post("/auth/token", response_model=TokenPair, tags=["Auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> TokenPair:
    """
    OAuth2 password flow. Clients send username+password, receive JWT pair.
    In /docs: click "Authorize" and use these credentials to get a token.
    """
    user = _users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Account is deactivated")

    return TokenPair(
        access_token=create_access_token(user),
        refresh_token=create_refresh_token(user),
    )


# ── LESSON 2: Refresh token rotation ─────────────────────────────────────────

@app.post("/auth/refresh", response_model=TokenPair, tags=["Auth"])
async def refresh_tokens(body: RefreshRequest) -> TokenPair:
    """
    Refresh token rotation:
    1. Validate the refresh token
    2. IMMEDIATELY delete (invalidate) the used token
    3. Issue a brand new token pair
    This means stolen refresh tokens can only be used once.
    """
    token_hash = hashlib.sha256(body.refresh_token.encode()).hexdigest()
    token_data = _refresh_tokens.pop(token_hash, None)  # pop = delete on read

    if not token_data:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    if datetime.now(timezone.utc) > token_data["expires_at"]:
        raise HTTPException(status_code=401, detail="Refresh token expired")

    user = _users_db.get(token_data["user_email"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    # Issue a completely new token pair
    return TokenPair(
        access_token=create_access_token(user),
        refresh_token=create_refresh_token(user),
    )


@app.post("/auth/logout", tags=["Auth"])
async def logout(body: RefreshRequest) -> dict[str, str]:
    """Invalidates the refresh token (access tokens expire naturally)."""
    token_hash = hashlib.sha256(body.refresh_token.encode()).hexdigest()
    _refresh_tokens.pop(token_hash, None)
    return {"message": "Logged out successfully"}


# ── LESSON 3: Scope-protected routes ─────────────────────────────────────────

@app.get("/users/me", response_model=UserPublic, tags=["Users"])
async def get_me(
    user: StoredUser = Security(get_current_user, scopes=["users:read"]),
) -> UserPublic:
    """Requires 'users:read' scope. All roles have this."""
    return UserPublic(**user.model_dump())


@app.get("/users", response_model=list[UserPublic], tags=["Users"])
async def list_users(
    user: StoredUser = Security(get_current_user, scopes=["users:read", "admin:all"]),
) -> list[UserPublic]:
    """Requires BOTH 'users:read' AND 'admin:all' — only admin role has this."""
    return [UserPublic(**u.model_dump()) for u in _users_db.values()]


@app.delete("/users/{email}", tags=["Users"])
async def delete_user(
    email: str,
    _: StoredUser = Depends(require_role(Role.ADMIN)),
) -> dict[str, str]:
    """Role-based guard — only ADMIN role can delete users."""
    if email not in _users_db:
        raise HTTPException(status_code=404, detail="User not found")
    del _users_db[email]
    return {"message": f"User {email} deleted"}


# ── LESSON 4: API Key management ──────────────────────────────────────────────

@app.post("/api-keys", response_model=APIKeyResponse, status_code=201, tags=["API Keys"])
async def create_api_key(
    data: APIKeyCreate,
    user: StoredUser = Security(get_current_user, scopes=["agents:manage"]),
) -> APIKeyResponse:
    """
    Issue an API key for an agent or service.
    The plain key is shown ONCE. Only the hash is stored.
    """
    plain_key = f"sk_{secrets.token_urlsafe(40)}"
    key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

    _api_keys[key_hash] = {
        "user_email": user.email,
        "scopes": data.scopes,
        "name": data.name,
        "created_at": datetime.now(timezone.utc),
    }

    return APIKeyResponse(
        key=plain_key,  # shown ONCE
        name=data.name,
        scopes=data.scopes,
        created_at=datetime.now(timezone.utc),
    )


@app.get("/agent/tools", tags=["Agent"])
async def list_agent_tools(
    user: StoredUser = Depends(get_api_key_user),
) -> dict[str, Any]:
    """
    Route accessible via X-API-Key header.
    Try: create an API key above, then use it here in the X-API-Key header.
    """
    if not user:
        raise HTTPException(status_code=401, detail="API key required")
    return {
        "tools": ["search", "memory_read", "memory_write"],
        "agent": user.username,
        "tenant": user.tenant_id,
    }


# ── LESSON 5: Tenant-aware route ─────────────────────────────────────────────

@app.get("/tenant/data", tags=["Multi-tenant"])
async def get_tenant_data(
    ctx: TenantContext = Depends(get_tenant_context),
) -> dict[str, Any]:
    """
    All data is automatically scoped to the authenticated user's tenant.
    In a real app: pass ctx.tenant_id to every DB query.
    """
    # Simulate tenant-scoped data
    tenant_users = [
        u for u in _users_db.values()
        if u.tenant_id == ctx.tenant_id
    ]
    return {
        "tenant_id": ctx.tenant_id,
        "your_role": ctx.user.role,
        "tenant_user_count": len(tenant_users),
        "note": "All queries in a real app would filter by tenant_id",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRACTICE EXERCISES
# ─────────────────────────────────────────────────────────────────────────────
# 1. Add email verification: after registration, generate a verification token.
# 2. Implement JWT blacklisting for logout (store revoked JTIs in Redis/dict).
# 3. Add rate limiting on /auth/token to prevent brute force (5 attempts/min).
# 4. Add API key expiry: keys expire after 90 days.
# 5. Add scope validation when creating API keys (can't grant scopes you don't have).
