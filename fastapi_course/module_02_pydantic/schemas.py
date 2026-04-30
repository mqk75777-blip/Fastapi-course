"""
MODULE 2: Pydantic v2 — Data Contracts
=======================================
- Request vs response schemas (never expose DB models)
- Field validators, model_validators
- Discriminated unions, custom types
- LLM output validation (strict mode)
- Nested models, recursive schemas
"""

from __future__ import annotations
from datetime import datetime
from typing import Annotated, Any, Literal, Union
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
    computed_field,
)


# ─── Base pattern: separate Request / Response models ─────────────────────────
# NEVER return ORM models directly. Always use response schemas.

class UserCreate(BaseModel):
    """What the client sends to create a user."""
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=2, max_length=100)
    role: Literal["admin", "user", "viewer"] = "user"

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

    @field_validator("full_name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        return v.strip().title()


class UserResponse(BaseModel):
    """What we return to clients — never includes password."""
    model_config = ConfigDict(from_attributes=True)  # allows ORM → Pydantic

    id: UUID
    email: EmailStr
    full_name: str
    role: str
    is_active: bool
    created_at: datetime

    @computed_field
    @property
    def display_name(self) -> str:
        return self.full_name.split()[0]


class UserUpdate(BaseModel):
    """All fields optional for PATCH operations."""
    full_name: str | None = Field(None, min_length=2, max_length=100)
    role: Literal["admin", "user", "viewer"] | None = None


# ─── Nested models ────────────────────────────────────────────────────────────

class AddressSchema(BaseModel):
    street: str
    city: str
    country: str = "PK"
    postal_code: str | None = None


class CompanyCreate(BaseModel):
    name: str = Field(min_length=2, max_length=200)
    email: EmailStr
    address: AddressSchema
    employees: list[UserCreate] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def check_employees_have_unique_emails(self) -> CompanyCreate:
        emails = [e.email for e in self.employees]
        if len(emails) != len(set(emails)):
            raise ValueError("Employee emails must be unique")
        return self


# ─── Discriminated unions (for polymorphic API contracts) ─────────────────────

class TextMessage(BaseModel):
    type: Literal["text"] = "text"
    content: str = Field(min_length=1, max_length=4096)


class ImageMessage(BaseModel):
    type: Literal["image"] = "image"
    url: str
    alt_text: str | None = None


class ToolCallMessage(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_name: str
    arguments: dict[str, Any]


# Discriminated union — FastAPI picks the right model based on "type" field
Message = Annotated[
    Union[TextMessage, ImageMessage, ToolCallMessage],
    Field(discriminator="type")
]


class ConversationCreate(BaseModel):
    messages: list[Message]
    session_id: UUID = Field(default_factory=uuid4)


# ─── LLM Output Validation (strict mode) ─────────────────────────────────────
# Use this to validate and sanitize JSON returned by LLMs before storing/using

class LLMStructuredOutput(BaseModel):
    """
    Strict schema for validating LLM JSON output.
    If LLM returns garbage, this raises ValidationError — catch it and retry.
    """
    model_config = ConfigDict(strict=True)  # no type coercion

    intent: Literal["book_flight", "check_weather", "cancel_booking", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    entities: dict[str, str] = Field(default_factory=dict)
    requires_clarification: bool
    follow_up_question: str | None = None

    @field_validator("entities")
    @classmethod
    def sanitize_entities(cls, v: dict) -> dict:
        # Remove any keys/values that are too long (LLM hallucination guard)
        return {
            k[:50]: str(val)[:200]
            for k, val in v.items()
            if isinstance(k, str) and k.strip()
        }


def parse_llm_output(raw_json: dict) -> LLMStructuredOutput | None:
    """
    Safe wrapper to parse LLM output.
    Returns None if validation fails (caller should retry with better prompt).
    """
    try:
        return LLMStructuredOutput.model_validate(raw_json)
    except Exception as e:
        print(f"LLM output validation failed: {e}")
        return None


# ─── Generic paginated response wrapper ───────────────────────────────────────

from typing import TypeVar, Generic

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
    has_next: bool

    @computed_field
    @property
    def total_pages(self) -> int:
        return (self.total + self.page_size - 1) // self.page_size


# ─── API response envelope ────────────────────────────────────────────────────

class APIResponse(BaseModel, Generic[T]):
    """Standard response envelope for all API responses."""
    success: bool = True
    data: T | None = None
    message: str | None = None
    request_id: str | None = None


# Usage examples:
# return APIResponse(data=UserResponse(...), message="User created")
# return APIResponse[PaginatedResponse[UserResponse]](data=paginated_users)
