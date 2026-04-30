"""
MODULE 5: Database Patterns — SQLAlchemy Async + Alembic + pgvector
====================================================================
- SQLAlchemy 2.0 async ORM models
- Repository pattern (business logic never touches DB directly)
- Alembic migration setup
- pgvector for embedding storage (RAG / agent memory)
- Redis for caching
- Soft deletes, timestamps, audit log
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Generic, Sequence, TypeVar
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean, DateTime, ForeignKey, Index, String, Text, func, select, update
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ─── Base model with common fields ────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False
    )


class SoftDeleteMixin:
    """Soft deletes: records are never hard-deleted, just marked deleted."""
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    def soft_delete(self) -> None:
        self.deleted_at = datetime.now(timezone.utc)


# ─── ORM Models ───────────────────────────────────────────────────────────────

class User(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="user", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    conversations: Mapped[list[Conversation]] = relationship(back_populates="user")
    api_keys: Mapped[list[ApiKey]] = relationship(back_populates="user")


class ApiKey(Base, TimestampMixin):
    __tablename__ = "api_keys"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="api_keys")


class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str | None] = mapped_column(String(200))
    metadata: Mapped[dict] = mapped_column(JSONB, default=dict)

    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list[Message]] = relationship(back_populates="conversation", order_by="Message.created_at")


class Message(Base, TimestampMixin):
    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20), nullable=False)   # user | assistant | system | tool
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column()
    tool_calls: Mapped[dict | None] = mapped_column(JSONB)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class EmbeddingDocument(Base, TimestampMixin):
    """
    pgvector table — stores text chunks + their vector embeddings.
    Used for RAG (retrieval-augmented generation) and agent memory.
    """
    __tablename__ = "embedding_documents"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))  # OpenAI ada-002 dim
    source: Mapped[str | None] = mapped_column(String(255))
    metadata: Mapped[dict] = mapped_column(JSONB, default=dict)

    __table_args__ = (
        Index(
            "ix_embedding_documents_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


# ─── Generic base repository ──────────────────────────────────────────────────

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    def __init__(self, model: type[ModelType], db: AsyncSession):
        self.model = model
        self.db = db

    async def get(self, id: UUID) -> ModelType | None:
        result = await self.db.execute(select(self.model).where(self.model.id == id))
        return result.scalar_one_or_none()

    async def get_all(self, skip: int = 0, limit: int = 50) -> Sequence[ModelType]:
        result = await self.db.execute(select(self.model).offset(skip).limit(limit))
        return result.scalars().all()

    async def create(self, obj_in: dict) -> ModelType:
        db_obj = self.model(**obj_in)
        self.db.add(db_obj)
        await self.db.flush()  # get ID without committing
        await self.db.refresh(db_obj)
        return db_obj

    async def update(self, id: UUID, obj_in: dict) -> ModelType | None:
        await self.db.execute(
            update(self.model).where(self.model.id == id).values(**obj_in)
        )
        return await self.get(id)

    async def delete(self, id: UUID) -> bool:
        obj = await self.get(id)
        if not obj:
            return False
        await self.db.delete(obj)
        return True


# ─── User repository ──────────────────────────────────────────────────────────

class UserRepository(BaseRepository[User]):
    def __init__(self, db: AsyncSession):
        super().__init__(User, db)

    async def get_by_email(self, email: str) -> User | None:
        result = await self.db.execute(
            select(User).where(User.email == email, User.deleted_at.is_(None))
        )
        return result.scalar_one_or_none()

    async def get_active_users(self, skip: int = 0, limit: int = 50) -> Sequence[User]:
        result = await self.db.execute(
            select(User)
            .where(User.is_active == True, User.deleted_at.is_(None))
            .offset(skip).limit(limit)
        )
        return result.scalars().all()

    async def soft_delete(self, user_id: UUID) -> bool:
        user = await self.get(user_id)
        if not user:
            return False
        user.soft_delete()
        return True


# ─── Embedding / vector search repository ────────────────────────────────────

class EmbeddingRepository(BaseRepository[EmbeddingDocument]):
    def __init__(self, db: AsyncSession):
        super().__init__(EmbeddingDocument, db)

    async def similarity_search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        similarity_threshold: float = 0.7,
    ) -> list[dict]:
        """
        Cosine similarity search using pgvector.
        Returns top-k most similar documents.
        """
        from sqlalchemy import text

        result = await self.db.execute(
            text("""
                SELECT id, content, source, metadata,
                       1 - (embedding <=> :embedding::vector) AS similarity
                FROM embedding_documents
                WHERE 1 - (embedding <=> :embedding::vector) >= :threshold
                ORDER BY embedding <=> :embedding::vector
                LIMIT :limit
            """),
            {
                "embedding": str(query_embedding),
                "threshold": similarity_threshold,
                "limit": limit,
            }
        )
        rows = result.fetchall()
        return [
            {"id": str(r.id), "content": r.content, "source": r.source, "similarity": round(r.similarity, 4)}
            for r in rows
        ]

    async def upsert_document(
        self,
        content: str,
        embedding: list[float],
        source: str,
        metadata: dict | None = None,
    ) -> EmbeddingDocument:
        doc = EmbeddingDocument(
            content=content,
            embedding=embedding,
            source=source,
            metadata=metadata or {},
        )
        self.db.add(doc)
        await self.db.flush()
        return doc


# ─── Conversation repository ───────────────────────────────────────────────────

class ConversationRepository(BaseRepository[Conversation]):
    def __init__(self, db: AsyncSession):
        super().__init__(Conversation, db)

    async def get_with_messages(self, conversation_id: UUID) -> Conversation | None:
        from sqlalchemy.orm import selectinload
        result = await self.db.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .options(selectinload(Conversation.messages))
        )
        return result.scalar_one_or_none()

    async def add_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        tool_calls: dict | None = None,
    ) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
        )
        self.db.add(msg)
        await self.db.flush()
        return msg


# ─── Alembic setup (alembic/env.py key parts) ─────────────────────────────────
"""
# alembic/env.py

from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context
from app.models import Base  # import all models here
from app.config import settings

config = context.config
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
target_metadata = Base.metadata

async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()

# Commands:
# alembic init -t async alembic
# alembic revision --autogenerate -m "initial"
# alembic upgrade head
# alembic downgrade -1
"""
