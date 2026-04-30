"""
MODULE 4: Auth, Security & Middleware
======================================
- JWT access + refresh token flow
- API key auth for machine-to-machine calls
- Role-based access control (RBAC) via DI
- Rate limiting with slowapi
- Security middleware: request ID, HTTPS
"""

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

import jwt
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import settings

router = APIRouter()


# ─── Password hashing ─────────────────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ─── JWT token creation & validation ──────────────────────────────────────────

class TokenPayload(BaseModel):
    sub: str          # user ID
    role: str
    type: Literal["access", "refresh"]
    exp: datetime
    jti: str          # unique token ID (for blacklisting)


def create_access_token(user_id: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {
        "sub": user_id,
        "role": role,
        "type": "access",
        "exp": expire,
        "jti": str(uuid.uuid4()),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(user_id: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.REFRESH_TOKEN_EXPIRE_DAYS
    )
    payload = {
        "sub": user_id,
        "role": role,
        "type": "refresh",
        "exp": expire,
        "jti": str(uuid.uuid4()),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─── Current user dependency ──────────────────────────────────────────────────

security = HTTPBearer()


class CurrentUser(BaseModel):
    id: str
    role: str
    token_jti: str


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> CurrentUser:
    payload = decode_token(credentials.credentials)

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh tokens cannot be used for API access"
        )

    # TODO: check token JTI against Redis blacklist (Module 5)
    # if await redis.get(f"blacklist:{payload['jti']}"):
    #     raise HTTPException(401, "Token has been revoked")

    return CurrentUser(
        id=payload["sub"],
        role=payload["role"],
        token_jti=payload["jti"],
    )


AuthUser = Annotated[CurrentUser, Depends(get_current_user)]


# ─── RBAC — role-based access control ────────────────────────────────────────

def require_role(*roles: str):
    """
    Dependency factory. Usage:
        @router.delete("/users/{id}", dependencies=[Depends(require_role("admin"))])
    """
    async def check_role(current_user: AuthUser) -> CurrentUser:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {roles}. Your role: {current_user.role}"
            )
        return current_user
    return check_role


AdminRequired = Depends(require_role("admin"))
AdminOrUser = Depends(require_role("admin", "user"))


# ─── API key auth (for agents/machines calling your API) ──────────────────────

def hash_api_key(key: str) -> str:
    """Store only the hash in DB, never the raw key."""
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Returns (raw_key, hashed_key). Store hash, give raw to client."""
    raw = f"sk-{secrets.token_urlsafe(32)}"
    return raw, hash_api_key(raw)


async def get_api_key_user(
    x_api_key: Annotated[str | None, Header()] = None
) -> dict | None:
    """
    Used for machine-to-machine (agent ↔ API) calls.
    Returns None if no key provided — combine with JWT auth for mixed endpoints.
    """
    if not x_api_key:
        return None

    hashed = hash_api_key(x_api_key)

    # TODO: lookup hashed key in DB
    # api_key_record = await db.query(ApiKey).filter_by(key_hash=hashed).first()
    # if not api_key_record or not api_key_record.is_active:
    #     raise HTTPException(401, "Invalid API key")
    # return api_key_record.owner

    # Stub for now:
    if hashed:  # always true — replace with real DB check
        return {"id": "machine-user", "role": "user", "type": "api_key"}
    raise HTTPException(status_code=401, detail="Invalid API key")


# ─── Rate limiting ────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# In main.py add:
# from slowapi import _rate_limit_exceeded_handler
# from slowapi.errors import RateLimitExceeded
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ─── Auth endpoints ───────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60


@router.post("/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    # TODO: lookup user in DB, verify password
    # user = await user_repo.get_by_email(body.email)
    # if not user or not verify_password(body.password, user.hashed_password):
    #     raise HTTPException(401, "Invalid credentials")

    # Stub:
    fake_user_id = "user-123"
    fake_role = "admin"

    return TokenResponse(
        access_token=create_access_token(fake_user_id, fake_role),
        refresh_token=create_refresh_token(fake_user_id, fake_role),
    )


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/auth/refresh", response_model=TokenResponse)
async def refresh_tokens(body: RefreshRequest):
    payload = decode_token(body.refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=400, detail="Not a refresh token")

    # TODO: blacklist old refresh token JTI in Redis
    # await redis.setex(f"blacklist:{payload['jti']}", REFRESH_TOKEN_EXPIRE_DAYS * 86400, "1")

    return TokenResponse(
        access_token=create_access_token(payload["sub"], payload["role"]),
        refresh_token=create_refresh_token(payload["sub"], payload["role"]),
    )


@router.post("/auth/logout")
async def logout(current_user: AuthUser):
    # Blacklist the current access token JTI in Redis
    # await redis.setex(f"blacklist:{current_user.token_jti}", ACCESS_TOKEN_EXPIRE_MINUTES * 60, "1")
    return {"message": "Logged out successfully"}


@router.get("/users/me")
async def get_me(current_user: AuthUser):
    return current_user


@router.delete("/admin/users/{user_id}", dependencies=[AdminRequired])
async def delete_user(user_id: str, current_user: AuthUser):
    return {"message": f"User {user_id} deleted by {current_user.id}"}


@router.post("/api-keys/generate")
async def create_api_key(current_user: AuthUser):
    raw_key, hashed_key = generate_api_key()
    # TODO: store hashed_key in DB linked to current_user.id
    return {
        "api_key": raw_key,
        "message": "Store this key — it won't be shown again",
    }
