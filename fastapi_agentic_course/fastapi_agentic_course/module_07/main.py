"""
================================================================================
MODULE 07 — Testing — Professional Grade
================================================================================
Topics:
  L1. pytest-asyncio + httpx test client setup
  L2. Mocking LLM API calls with respx
  L3. Integration tests with test database
  L4. Contract testing for agent tool APIs
  L5. Coverage & CI/CD

Run tests:
  pytest module_07/ -v
  pytest module_07/ -v --cov=module_07 --cov-report=term-missing

This file contains BOTH the app and the tests (for simplicity).
In a real project: app/ and tests/ are separate directories.
================================================================================
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# THE APP UNDER TEST
# ─────────────────────────────────────────────────────────────────────────────

class ProductCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    price: float = Field(gt=0)


class Product(BaseModel):
    id: str
    name: str
    price: float


class LLMSummaryRequest(BaseModel):
    text: str


class LLMSummaryResponse(BaseModel):
    summary: str
    model: str
    tokens_used: int


# Simulated external LLM client
class GroqClient:
    def __init__(self, api_key: str = "test-key") -> None:
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"

    async def summarize(self, text: str) -> LLMSummaryResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "llama3-8b-8192",
                    "messages": [
                        {"role": "system", "content": "Summarize the following text in one sentence."},
                        {"role": "user", "content": text},
                    ],
                    "max_tokens": 200,
                },
            )
            response.raise_for_status()
            data = response.json()
            return LLMSummaryResponse(
                summary=data["choices"][0]["message"]["content"],
                model=data["model"],
                tokens_used=data["usage"]["total_tokens"],
            )


# In-memory store
_products: dict[str, Product] = {}
_groq_client = GroqClient()


def get_groq_client() -> GroqClient:
    return _groq_client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _products.clear()
    # Seed data
    p = Product(id="prod-001", name="FastAPI Book", price=29.99)
    _products[p.id] = p
    yield


app = FastAPI(title="App Under Test", lifespan=lifespan)


@app.get("/products", response_model=list[Product], tags=["Products"])
async def list_products() -> list[Product]:
    return list(_products.values())


@app.get("/products/{product_id}", response_model=Product, tags=["Products"])
async def get_product(product_id: str) -> Product:
    product = _products.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@app.post("/products", response_model=Product, status_code=201, tags=["Products"])
async def create_product(data: ProductCreate) -> Product:
    product = Product(id=str(uuid4()), **data.model_dump())
    _products[product.id] = product
    return product


@app.delete("/products/{product_id}", tags=["Products"])
async def delete_product(product_id: str) -> dict[str, str]:
    if product_id not in _products:
        raise HTTPException(status_code=404, detail="Product not found")
    del _products[product_id]
    return {"message": "Deleted"}


@app.post("/summarize", response_model=LLMSummaryResponse, tags=["AI"])
async def summarize_text(
    data: LLMSummaryRequest,
    groq: GroqClient = Depends(get_groq_client),
) -> LLMSummaryResponse:
    """Calls Groq API to summarize text. We will mock this in tests."""
    try:
        return await groq.summarize(data.text)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {e}")


# Agent tool endpoint — used in contract testing (L4)
class ToolInput(BaseModel):
    query: str = Field(description="Search query", min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=100)


class ToolOutput(BaseModel):
    results: list[dict[str, Any]]
    total: int
    query: str


@app.post("/tools/search", response_model=ToolOutput, tags=["Agent Tools"])
async def search_tool(data: ToolInput) -> ToolOutput:
    """Agent tool — strict input/output contract."""
    matching = [p.model_dump() for p in _products.values() if data.query.lower() in p.name.lower()]
    return ToolOutput(results=matching[:data.limit], total=len(matching), query=data.query)


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 1 — Test Setup & Fixtures
# ─────────────────────────────────────────────────────────────────────────────
# pytest fixtures provide shared setup/teardown across tests.
# scope="session" — created once for all tests
# scope="function" — created fresh for each test (default, safest for state)

@pytest.fixture(scope="session")
def sync_client():
    """
    LESSON 1 — Synchronous test client.
    Use this for simple request/response tests.
    TestClient manages the lifespan automatically.
    """
    with TestClient(app) as client:
        yield client


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """
    LESSON 1 — Async test client.
    Use when your test itself needs to be async (concurrent requests, WebSockets).
    """
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture(autouse=True)
def reset_products():
    """
    LESSON 1 — State isolation.
    autouse=True: runs for EVERY test automatically.
    Resets the in-memory store before each test to prevent state leakage.
    """
    _products.clear()
    _products["prod-001"] = Product(id="prod-001", name="FastAPI Book", price=29.99)
    yield
    # Teardown (optional cleanup after test)


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 1 — Basic CRUD Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProductCRUD:
    """Group related tests in a class — no shared state between test methods."""

    def test_list_products_returns_200(self, sync_client: TestClient) -> None:
        response = sync_client.get("/products")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1  # seeded product

    def test_get_existing_product(self, sync_client: TestClient) -> None:
        response = sync_client.get("/products/prod-001")
        assert response.status_code == 200
        product = response.json()
        assert product["id"] == "prod-001"
        assert product["name"] == "FastAPI Book"
        assert product["price"] == 29.99

    def test_get_nonexistent_product_returns_404(self, sync_client: TestClient) -> None:
        response = sync_client.get("/products/does-not-exist")
        assert response.status_code == 404
        # Check the error response SHAPE — clients/agents depend on this structure
        error = response.json()
        assert "detail" in error

    def test_create_product_returns_201(self, sync_client: TestClient) -> None:
        payload = {"name": "Python Handbook", "price": 39.99}
        response = sync_client.post("/products", json=payload)
        assert response.status_code == 201
        product = response.json()
        assert product["name"] == "Python Handbook"
        assert product["price"] == 39.99
        assert "id" in product  # auto-generated

    def test_create_product_validates_price(self, sync_client: TestClient) -> None:
        """Pydantic validation should reject negative prices."""
        response = sync_client.post("/products", json={"name": "Bad", "price": -1})
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any("price" in str(e) for e in errors)

    def test_create_product_validates_empty_name(self, sync_client: TestClient) -> None:
        response = sync_client.post("/products", json={"name": "", "price": 10.0})
        assert response.status_code == 422

    def test_delete_existing_product(self, sync_client: TestClient) -> None:
        response = sync_client.delete("/products/prod-001")
        assert response.status_code == 200
        # Verify it's gone
        get_response = sync_client.get("/products/prod-001")
        assert get_response.status_code == 404

    def test_delete_nonexistent_returns_404(self, sync_client: TestClient) -> None:
        response = sync_client.delete("/products/ghost")
        assert response.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 2 — Mocking LLM API Calls
# ─────────────────────────────────────────────────────────────────────────────
# Never call real LLM APIs in tests:
# - Costs money on every test run
# - Tests become flaky (network failures)
# - Tests are slow
# Use dependency_overrides to inject a fake client, or respx to mock HTTP calls.

class FakeGroqClient(GroqClient):
    """Fake Groq client that returns predictable responses."""

    async def summarize(self, text: str) -> LLMSummaryResponse:
        # Deterministic response — always the same for the same input
        return LLMSummaryResponse(
            summary=f"SUMMARY: {text[:30]}...",
            model="llama3-8b-8192",
            tokens_used=42,
        )


@pytest.fixture
def client_with_fake_llm(sync_client: TestClient) -> TestClient:
    """Override the LLM dependency for the duration of a test."""
    app.dependency_overrides[get_groq_client] = lambda: FakeGroqClient()
    yield sync_client
    app.dependency_overrides.clear()


class TestLLMSummarization:

    def test_summarize_uses_fake_llm(self, client_with_fake_llm: TestClient) -> None:
        """Test the summarize endpoint without calling real Groq API."""
        response = client_with_fake_llm.post(
            "/summarize",
            json={"text": "FastAPI is a modern Python web framework for building APIs."},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["summary"].startswith("SUMMARY:")
        assert result["model"] == "llama3-8b-8192"
        assert result["tokens_used"] == 42  # from our fake

    def test_summarize_validates_empty_text(self, client_with_fake_llm: TestClient) -> None:
        response = client_with_fake_llm.post("/summarize", json={"text": ""})
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 4 — Contract Testing for Agent Tool APIs
# ─────────────────────────────────────────────────────────────────────────────
# Contract tests verify that your API's schema NEVER breaks agent integrations.
# If an agent was trained to expect {"results": [...], "total": N},
# your API must always return exactly that shape.

class TestAgentToolContracts:
    """
    Contract tests for the /tools/search endpoint.
    These tests verify the SCHEMA contract, not just behavior.
    If any of these break, agent integrations will break.
    """

    def test_search_tool_returns_expected_schema(self, sync_client: TestClient) -> None:
        response = sync_client.post("/tools/search", json={"query": "Book", "limit": 10})
        assert response.status_code == 200

        data = response.json()

        # CONTRACT: these fields MUST always be present
        assert "results" in data, "Contract violation: missing 'results' field"
        assert "total" in data, "Contract violation: missing 'total' field"
        assert "query" in data, "Contract violation: missing 'query' field"

        # CONTRACT: types must be stable
        assert isinstance(data["results"], list), "Contract violation: 'results' must be array"
        assert isinstance(data["total"], int), "Contract violation: 'total' must be integer"
        assert isinstance(data["query"], str), "Contract violation: 'query' must be string"

    def test_search_tool_validates_input_schema(self, sync_client: TestClient) -> None:
        """Agents must send valid input — verify rejection of bad requests."""
        # Missing required field
        response = sync_client.post("/tools/search", json={})
        assert response.status_code == 422

        # Limit exceeds maximum
        response = sync_client.post("/tools/search", json={"query": "test", "limit": 9999})
        assert response.status_code == 422

        # Empty query
        response = sync_client.post("/tools/search", json={"query": ""})
        assert response.status_code == 422

    def test_search_tool_result_items_have_stable_schema(self, sync_client: TestClient) -> None:
        """Each result item must have a predictable structure."""
        response = sync_client.post("/tools/search", json={"query": "Book"})
        data = response.json()

        for item in data["results"]:
            # These fields are part of the agent tool contract
            assert "id" in item, "Result items must have 'id'"
            assert "name" in item, "Result items must have 'name'"
            assert "price" in item, "Result items must have 'price'"

    def test_search_tool_respects_limit(self, sync_client: TestClient) -> None:
        # Add more products
        for i in range(5):
            _products[f"extra-{i}"] = Product(id=f"extra-{i}", name=f"Book {i}", price=10.0)

        response = sync_client.post("/tools/search", json={"query": "Book", "limit": 2})
        data = response.json()
        assert len(data["results"]) <= 2


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 1 — Async Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAsyncConcurrency:
    """Tests that verify concurrent behavior — requires async test client."""

    async def test_concurrent_requests(self, async_client: httpx.AsyncClient) -> None:
        """Simulate multiple simultaneous clients."""
        import asyncio

        tasks = [async_client.get("/products") for _ in range(10)]
        responses = await asyncio.gather(*tasks)

        for response in responses:
            assert response.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# LESSON 5 — pytest.ini / pyproject.toml config
# ─────────────────────────────────────────────────────────────────────────────
# Add to pyproject.toml:
#
# [tool.pytest.ini_options]
# asyncio_mode = "auto"
# testpaths = ["tests"]
# addopts = "--cov=app --cov-report=term-missing --cov-fail-under=80"
#
# [tool.coverage.run]
# omit = ["*/migrations/*", "*/tests/*"]
#
# Run:
#   pytest -v                          # all tests with verbose output
#   pytest -v -k "test_contract"       # only contract tests
#   pytest --cov=. --cov-report=html   # coverage with HTML report

# ─────────────────────────────────────────────────────────────────────────────
# GITHUB ACTIONS CI (Lesson 5)
# ─────────────────────────────────────────────────────────────────────────────
# Save as .github/workflows/ci.yml:
#
# name: CI
# on: [push, pull_request]
# jobs:
#   test:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v4
#       - uses: actions/setup-python@v5
#         with: {python-version: "3.12"}
#       - run: pip install -r requirements.txt
#       - run: pytest --cov=. --cov-fail-under=80
#       - run: pip install ruff mypy
#       - run: ruff check .
#       - run: mypy .

# ─────────────────────────────────────────────────────────────────────────────
# PRACTICE EXERCISES
# ─────────────────────────────────────────────────────────────────────────────
# 1. Write a test that verifies the LLM error (502) when Groq returns 500.
# 2. Add a parametrize test for create_product with multiple price/name combos.
# 3. Write a test that checks the response time is under 200ms (performance test).
# 4. Add a snapshot test that saves the tool output schema and alerts if it changes.
# 5. Write a load test using asyncio.gather with 100 concurrent requests.
