"""
================================================================================
MODULE 02 — Pydantic v2 Mastery
================================================================================
Topics:
  L1. field_validator, model_validator, computed_field
  L2. Custom types with Annotated
  L3. Response models & schema design for AI tool calling
  L4. Strict mode, discriminated unions, tagged unions

Run:  uvicorn main:app --reload
Docs: http://localhost:8000/docs
================================================================================
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from pydantic import (
    UUID4,
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    EmailStr,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic.functional_validators import ModelBeforeValidator


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 1 — field_validator & model_validator
# ─────────────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    """
    Demonstrates field-level and model-level validators.
    In Pydantic v2, validators use decorators — cleaner than v1's @validator.
    """

    model_config = ConfigDict(
        # str_strip_whitespace: automatically strips leading/trailing whitespace
        str_strip_whitespace=True,
        # str_to_lower: useful for emails, usernames
        # strict: False means Pydantic will coerce types (e.g. "123" → 123)
        populate_by_name=True,  # allow using field name OR alias
    )

    username: str = Field(min_length=3, max_length=50, examples=["qasim_dev"])
    email: EmailStr
    password: str = Field(min_length=8, exclude=True)  # exclude=True: never in output
    confirm_password: str = Field(exclude=True)
    age: int = Field(ge=13, le=120)
    phone: str | None = None

    # ── Field validator (runs on a single field) ──────────────────────────────
    @field_validator("username")
    @classmethod
    def username_must_be_alphanumeric(cls, v: str) -> str:
        """
        mode='after' (default): runs after Pydantic's own type coercion.
        Receives the already-converted value — safe to call .lower(), etc.
        """
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("Username must contain only letters, numbers, underscores")
        return v.lower()  # normalize to lowercase

    @field_validator("phone")
    @classmethod
    def normalize_phone(cls, v: str | None) -> str | None:
        if v is None:
            return None
        # Strip spaces and dashes, keep only digits and leading +
        cleaned = re.sub(r"[\s\-\(\)]", "", v)
        if not re.match(r"^\+?[0-9]{7,15}$", cleaned):
            raise ValueError("Invalid phone number format")
        return cleaned

    # ── Model validator (runs on the whole model) ─────────────────────────────
    @model_validator(mode="after")
    def passwords_must_match(self) -> UserCreate:
        """
        mode='after': runs after ALL field validators pass.
        'self' is the fully-constructed model instance.
        Use for cross-field validation.
        """
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self

    # ── Computed field (derived, always up-to-date) ───────────────────────────
    @computed_field
    @property
    def display_name(self) -> str:
        """Computed fields are included in serialization automatically."""
        return f"@{self.username}"


class UserResponse(BaseModel):
    """
    Separate response model — never expose passwords or internal fields.
    Always have distinct input and output models in professional APIs.
    """
    id: UUID4
    username: str
    email: EmailStr
    display_name: str
    age: int
    phone: str | None
    created_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 2 — Custom types with Annotated
# ─────────────────────────────────────────────────────────────────────────────
# Annotated lets you attach validators to a type and reuse them everywhere.
# This is far better than copy-pasting @field_validator across models.

def _validate_positive(v: float) -> float:
    if v <= 0:
        raise ValueError("Must be positive")
    return v

def _normalize_currency(v: str) -> str:
    v = v.upper().strip()
    if v not in {"USD", "EUR", "GBP", "PKR", "CNY"}:
        raise ValueError(f"Unsupported currency: {v}")
    return v

def _coerce_to_int_percentage(v: Any) -> int:
    """BeforeValidator runs BEFORE Pydantic's type conversion."""
    if isinstance(v, float):
        return int(v)  # 99.9 → 99
    if isinstance(v, str) and v.endswith("%"):
        return int(v[:-1])  # "85%" → 85
    return v

# Reusable annotated types — use these as field types anywhere
PositiveFloat = Annotated[float, AfterValidator(_validate_positive)]
Currency = Annotated[str, AfterValidator(_normalize_currency)]
Percentage = Annotated[int, BeforeValidator(_coerce_to_int_percentage), Field(ge=0, le=100)]


class PricingPlan(BaseModel):
    name: str
    monthly_price: PositiveFloat         # reusable — validates > 0
    annual_price: PositiveFloat
    currency: Currency                   # reusable — normalizes & validates
    discount_percentage: Percentage      # reusable — coerces "10%" → 10

    @computed_field
    @property
    def annual_savings(self) -> float:
        """How much the user saves by paying annually vs monthly × 12."""
        return round((self.monthly_price * 12) - self.annual_price, 2)

    @computed_field
    @property
    def effective_monthly(self) -> float:
        """Annual plan cost per month."""
        return round(self.annual_price / 12, 2)


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 3 — Schema design for AI tool calling
# ─────────────────────────────────────────────────────────────────────────────
# When agents call your API as tools, the OpenAPI schema IS the tool definition.
# Well-designed schemas make agents more reliable — clear descriptions,
# constrained types, and examples reduce hallucinations.

class SearchFilters(BaseModel):
    """
    Detailed descriptions in Field() become the tool parameter descriptions
    that an AI agent reads to understand how to call your API correctly.
    """
    query: str = Field(
        description="Natural language search query",
        examples=["Pakistani traders importing from China"],
        min_length=1,
        max_length=500,
    )
    category: str | None = Field(
        default=None,
        description="Filter by product category. Use null to search all categories.",
        examples=["electronics", "textiles"],
    )
    min_price: float | None = Field(
        default=None,
        description="Minimum price in USD. Must be positive.",
        ge=0,
    )
    max_price: float | None = Field(
        default=None,
        description="Maximum price in USD. Must be greater than min_price.",
        ge=0,
    )
    page: int = Field(default=1, ge=1, description="Page number, starting at 1")
    page_size: int = Field(default=20, ge=1, le=100, description="Results per page (max 100)")

    @model_validator(mode="after")
    def validate_price_range(self) -> SearchFilters:
        if self.min_price and self.max_price:
            if self.min_price > self.max_price:
                raise ValueError("min_price must be less than max_price")
        return self


class SearchResult(BaseModel):
    """Structured, predictable response — agents parse this reliably."""
    id: UUID4
    name: str
    price: float
    currency: str
    category: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class PaginatedResponse(BaseModel):
    """
    Generic paginated response pattern.
    In production: use Generic[T] for full type safety.
    """
    items: list[SearchResult]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 4 — Discriminated unions for agentic message routing
# ─────────────────────────────────────────────────────────────────────────────
# Agents send different message types. Discriminated unions let you parse
# them into the correct model automatically using a "type" field.
# This is CRITICAL for building reliable agent message pipelines.

class TextMessage(BaseModel):
    type: Literal["text"]
    content: str
    role: Literal["user", "assistant", "system"]


class ToolCallMessage(BaseModel):
    type: Literal["tool_call"]
    tool_name: str
    tool_input: dict[str, Any]
    call_id: str = Field(default_factory=lambda: str(uuid4()))


class ToolResultMessage(BaseModel):
    type: Literal["tool_result"]
    call_id: str  # matches the ToolCallMessage.call_id
    result: Any
    is_error: bool = False


class ThinkingMessage(BaseModel):
    type: Literal["thinking"]
    content: str  # agent's internal monologue (Claude extended thinking)


# The discriminated union — Pydantic routes to the correct model based on "type"
# This is O(1) lookup via the discriminator field, not O(n) trial-and-error.
AgentMessage = Annotated[
    TextMessage | ToolCallMessage | ToolResultMessage | ThinkingMessage,
    Field(discriminator="type"),
]


class AgentConversation(BaseModel):
    """
    A full agent conversation with typed messages.
    Pydantic automatically picks the right model for each message in the list.
    """
    id: UUID4 = Field(default_factory=uuid4)
    agent_id: str
    messages: list[AgentMessage]
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        # In strict mode, Pydantic does NOT coerce types.
        # "1" will NOT become 1. True will NOT become 1.
        # Use strict mode for agent APIs where type mismatches indicate bugs.
        strict=False,  # flip to True and watch your tests catch type errors
    )


# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES — demonstrating all models above
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Module 02 — Pydantic v2 Mastery",
    description="Professional Pydantic patterns for agentic AI APIs",
    version="1.0.0",
)

# In-memory stores for demo
_users: dict[str, UserResponse] = {}
_conversations: dict[str, AgentConversation] = {}


@app.post("/users", response_model=UserResponse, status_code=201, tags=["Users"])
async def create_user(data: UserCreate) -> UserResponse:
    """
    Creates a user. Notice:
    - password/confirm_password are excluded from response (exclude=True)
    - display_name is computed and included automatically
    - response_model=UserResponse filters the output shape
    """
    user = UserResponse(
        id=uuid4(),
        username=data.username,
        email=data.email,
        display_name=data.display_name,
        age=data.age,
        phone=data.phone,
        created_at=datetime.utcnow(),
    )
    _users[str(user.id)] = user
    return user


@app.post("/pricing", response_model=PricingPlan, tags=["Pricing"])
async def create_pricing(data: PricingPlan) -> PricingPlan:
    """
    Demonstrates custom Annotated types and computed fields.
    Try: currency="usd" (gets normalized to "USD")
    Try: discount_percentage="15%" (gets coerced to 15)
    """
    return data


@app.post("/search", response_model=PaginatedResponse, tags=["Search"])
async def search_products(filters: SearchFilters) -> PaginatedResponse:
    """
    Demonstrates schema design for AI tool calling.
    Check /docs — the schema descriptions are exactly what an agent reads.
    """
    # Fake results for demonstration
    fake_items = [
        SearchResult(
            id=uuid4(),
            name=f"Product matching '{filters.query}' #{i}",
            price=round(10.0 * i, 2),
            currency="USD",
            category=filters.category or "general",
            relevance_score=round(1.0 - (i * 0.1), 2),
        )
        for i in range(1, min(filters.page_size + 1, 6))
    ]
    return PaginatedResponse(
        items=fake_items,
        total=50,
        page=filters.page,
        page_size=filters.page_size,
        total_pages=3,
        has_next=filters.page < 3,
        has_prev=filters.page > 1,
    )


@app.post("/conversations", response_model=AgentConversation, status_code=201, tags=["Agent"])
async def create_conversation(data: AgentConversation) -> AgentConversation:
    """
    Demonstrates discriminated unions.
    POST a conversation with mixed message types — Pydantic routes each one.

    Example body:
    {
      "agent_id": "agent-001",
      "messages": [
        {"type": "text", "content": "Search for laptops", "role": "user"},
        {"type": "tool_call", "tool_name": "search", "tool_input": {"query": "laptops"}},
        {"type": "tool_result", "call_id": "abc", "result": {"items": []}},
        {"type": "text", "content": "Here are the results...", "role": "assistant"}
      ]
    }
    """
    _conversations[str(data.id)] = data
    return data


@app.get("/conversations/{conversation_id}", response_model=AgentConversation, tags=["Agent"])
async def get_conversation(conversation_id: str) -> AgentConversation:
    conv = _conversations.get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.get("/schema/agent-message", tags=["Schema"])
async def agent_message_schema() -> dict[str, Any]:
    """
    Returns the JSON Schema for AgentMessage.
    This is what you'd feed to an LLM as a tool definition.
    """
    from pydantic import TypeAdapter
    adapter = TypeAdapter(AgentMessage)
    return adapter.json_schema()


# ─────────────────────────────────────────────────────────────────────────────
# PRACTICE EXERCISES
# ─────────────────────────────────────────────────────────────────────────────
# 1. Add a model_validator(mode='before') to UserCreate that trims all string fields.
# 2. Create an Annotated type for a Pakistani phone number (+92...).
# 3. Add a FileUploadMessage to AgentMessage discriminated union.
# 4. Make PaginatedResponse generic: PaginatedResponse[T] using Generic[T].
# 5. Add strict=True to AgentConversation and observe which tests break.
