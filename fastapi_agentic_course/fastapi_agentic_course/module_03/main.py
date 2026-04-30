"""
================================================================================
MODULE 03 — Async SQLAlchemy 2.0 & Database Patterns
================================================================================
Topics:
  L1. Async engine, session factory, connection pooling
  L2. Repository pattern with FastAPI DI
  L3. Transaction management & rollback
  L4. Alembic migrations (see migrations/ folder instructions)
  L5. Query optimization & N+1 prevention

Run:  uvicorn main:app --reload
      Uses SQLite (aiosqlite) for local dev. Swap to asyncpg for PostgreSQL.
Docs: http://localhost:8000/docs
================================================================================
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    delete,
    event,
    func,
    select,
    text,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, selectinload


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 1 — Async Engine & Session Factory
# ─────────────────────────────────────────────────────────────────────────────

# SQLite for local dev (zero setup required)
# In production swap to:
#   DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/dbname"
DATABASE_URL = "sqlite+aiosqlite:///./module03.db"

# create_async_engine — the connection pool
engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=True,          # logs every SQL query — disable in production
    pool_size=5,        # SQLite ignores this; critical for PostgreSQL
    max_overflow=10,    # extra connections above pool_size when under load
    pool_timeout=30,    # seconds to wait for a connection from the pool
    pool_pre_ping=True, # validates connections before handing them out
                        # prevents "connection reset" errors on idle connections
    # For serverless/Lambda: use NullPool (no persistent connections)
    # from sqlalchemy.pool import NullPool
    # poolclass=NullPool,
)

# async_sessionmaker — a factory that creates AsyncSession objects
# expire_on_commit=False: objects remain usable after session.commit()
# Without this, accessing an attribute after commit triggers a lazy load
# which fails in async context (no implicit IO in async SQLAlchemy).
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# MODELS — SQLAlchemy ORM (2.0 mapped_column syntax)
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Base class for all ORM models. Provides metadata registry."""
    pass


class UserORM(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationship — one user has many posts
    # lazy="raise" means SQLAlchemy will RAISE an error if you try to access
    # .posts without explicitly loading it. This prevents N+1 bugs silently.
    posts: Mapped[list[PostORM]] = relationship(
        "PostORM",
        back_populates="author",
        lazy="raise",  # LESSON 5: forces you to use explicit loading strategies
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_users_email_username", "email", "username"),  # composite index
    )


class PostORM(Base):
    __tablename__ = "posts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    title: Mapped[str] = mapped_column(String(200), index=True)
    body: Mapped[str] = mapped_column(Text)
    author_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    author: Mapped[UserORM] = relationship("UserORM", back_populates="posts", lazy="raise")


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: str
    password: str = Field(min_length=8)


class PostCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1)


class PostSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # allows ORM → Pydantic conversion
    id: str
    title: str
    body: str
    author_id: str
    created_at: datetime


class UserSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    username: str
    email: str
    created_at: datetime
    posts: list[PostSchema] = []


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 2 — Repository Pattern
# ─────────────────────────────────────────────────────────────────────────────
# The repository abstracts ALL database access.
# Routes never write SQL — they call repository methods.
# This makes testing trivial: swap the real repo for a fake one.

class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, user_id: str) -> UserORM | None:
        result = await self.session.execute(
            select(UserORM).where(UserORM.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> UserORM | None:
        result = await self.session.execute(
            select(UserORM).where(UserORM.email == email)
        )
        return result.scalar_one_or_none()

    async def get_all(self, limit: int = 20, offset: int = 0) -> list[UserORM]:
        result = await self.session.execute(
            select(UserORM).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def create(self, data: UserCreate) -> UserORM:
        # Never store plain passwords — this is a placeholder
        user = UserORM(
            username=data.username,
            email=data.email,
            hashed_password=f"hashed_{data.password}",  # use passlib in Module 04
        )
        self.session.add(user)
        await self.session.flush()  # flush sends SQL but doesn't commit
        await self.session.refresh(user)  # reload from DB to get DB-generated values
        return user

    async def delete(self, user_id: str) -> bool:
        result = await self.session.execute(
            delete(UserORM).where(UserORM.id == user_id)
        )
        return result.rowcount > 0


class PostRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_user_id(self, user_id: str) -> list[PostORM]:
        """
        LESSON 5 — Explicit loading prevents N+1.
        selectinload issues ONE extra query to load all related objects.
        Much better than: for user in users: await session.refresh(user.posts)
        """
        result = await self.session.execute(
            select(PostORM)
            .where(PostORM.author_id == user_id)
            .options(selectinload(PostORM.author))  # eager-load author in same query batch
        )
        return list(result.scalars().all())

    async def get_users_with_posts(self) -> list[UserORM]:
        """
        LESSON 5 — selectinload on a collection relationship.
        ONE query for users + ONE query for all their posts.
        vs. N+1: one query for users + N queries for each user's posts.
        """
        result = await self.session.execute(
            select(UserORM)
            .options(selectinload(UserORM.posts))  # one extra query, loads all posts
        )
        return list(result.unique().scalars().all())

    async def create(self, user_id: str, data: PostCreate) -> PostORM:
        post = PostORM(title=data.title, body=data.body, author_id=user_id)
        self.session.add(post)
        await self.session.flush()
        await self.session.refresh(post)
        return post


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 3 — Transaction Management
# ─────────────────────────────────────────────────────────────────────────────
# get_db is a generator dependency that manages the session lifecycle.
# yield gives the session to the route handler.
# After the route returns, the finally block ensures cleanup.

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session.
    The session is committed if no exception occurs, rolled back otherwise.

    Usage in routes:
        session: AsyncSession = Depends(get_db)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()  # commit on success
        except Exception:
            await session.rollback()  # rollback on any error
            raise


def get_user_repo(session: AsyncSession = Depends(get_db)) -> UserRepository:
    return UserRepository(session)


def get_post_repo(session: AsyncSession = Depends(get_db)) -> PostRepository:
    return PostRepository(session)


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — create tables on startup (use Alembic in production)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Create all tables (dev only — use Alembic migrations in production)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database tables created")
    yield
    await engine.dispose()  # close all connections in the pool
    print("✅ Database connections closed")


# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Module 03 — Async SQLAlchemy",
    description="Production-grade async database patterns",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/users", response_model=UserSchema, status_code=201, tags=["Users"])
async def create_user(
    data: UserCreate,
    repo: UserRepository = Depends(get_user_repo),
) -> Any:
    existing = await repo.get_by_email(data.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = await repo.create(data)
    return user


@app.get("/users", response_model=list[UserSchema], tags=["Users"])
async def list_users(
    limit: int = 20,
    offset: int = 0,
    repo: UserRepository = Depends(get_user_repo),
) -> Any:
    return await repo.get_all(limit=limit, offset=offset)


@app.get("/users/{user_id}", response_model=UserSchema, tags=["Users"])
async def get_user(
    user_id: str,
    repo: UserRepository = Depends(get_user_repo),
) -> Any:
    user = await repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.post("/users/{user_id}/posts", response_model=PostSchema, status_code=201, tags=["Posts"])
async def create_post(
    user_id: str,
    data: PostCreate,
    user_repo: UserRepository = Depends(get_user_repo),
    post_repo: PostRepository = Depends(get_post_repo),
) -> Any:
    """
    LESSON 3 — Transaction across multiple repositories.
    Both repos share the same session (FastAPI caches Depends() per request).
    If post creation fails, the user lookup is also rolled back (atomically).
    """
    user = await user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return await post_repo.create(user_id, data)


@app.get("/users/{user_id}/posts", response_model=list[PostSchema], tags=["Posts"])
async def get_user_posts(
    user_id: str,
    repo: PostRepository = Depends(get_post_repo),
) -> Any:
    return await repo.get_by_user_id(user_id)


@app.get("/users-with-posts", response_model=list[UserSchema], tags=["Posts"])
async def get_users_with_posts(
    repo: PostRepository = Depends(get_post_repo),
) -> Any:
    """
    LESSON 5 — N+1 prevention via selectinload.
    Watch the SQL logs (echo=True): exactly 2 queries regardless of user count.
    """
    return await repo.get_users_with_posts()


@app.get("/db/stats", tags=["System"])
async def db_stats(session: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Raw SQL execution example — for stats and reports."""
    result = await session.execute(text("SELECT COUNT(*) FROM users"))
    user_count = result.scalar()
    result2 = await session.execute(text("SELECT COUNT(*) FROM posts"))
    post_count = result2.scalar()
    return {"users": user_count, "posts": post_count}


# ─────────────────────────────────────────────────────────────────────────────
# ALEMBIC SETUP INSTRUCTIONS (Lesson 4)
# ─────────────────────────────────────────────────────────────────────────────
# Run these commands from the module_03/ directory:
#
#   alembic init migrations
#   # Edit migrations/env.py: set target_metadata = Base.metadata
#   # Edit migrations/env.py: set sqlalchemy.url in alembic.ini
#
#   # Generate a migration from your ORM models:
#   alembic revision --autogenerate -m "create users and posts tables"
#
#   # Apply migration:
#   alembic upgrade head
#
#   # Rollback one step:
#   alembic downgrade -1
#
#   # See migration history:
#   alembic history
#
# Production tip: always review autogenerated migrations before applying.
# Alembic can miss: renamed columns, complex index changes, partial indexes.

# ─────────────────────────────────────────────────────────────────────────────
# PRACTICE EXERCISES
# ─────────────────────────────────────────────────────────────────────────────
# 1. Add a Tag model and many-to-many relationship with Post.
# 2. Implement soft delete: add deleted_at column, filter in all queries.
# 3. Add a get_by_id that uses joinedload instead of selectinload — compare SQL.
# 4. Implement a bulk_create method using session.add_all().
# 5. Add pagination metadata to list endpoints (total count, has_next).
