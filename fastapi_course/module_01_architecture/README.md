# Module 1 — Project Architecture & Environment

## Production folder structure

```
app/
├── main.py                    # App factory + lifespan
├── config.py                  # PydanticSettings (all env vars here)
├── api/
│   └── v1/
│       ├── router.py          # Aggregates all sub-routers
│       └── endpoints/
│           ├── users.py
│           ├── items.py
│           └── agents.py
├── core/
│   ├── security.py            # JWT, hashing (Module 4)
│   ├── dependencies.py        # Shared Depends() (Module 3)
│   └── exceptions.py          # Custom exception handlers (Module 6)
├── models/
│   └── user.py                # SQLAlchemy ORM models (Module 5)
├── schemas/
│   └── user.py                # Pydantic request/response schemas (Module 2)
├── services/
│   └── user_service.py        # Business logic layer
├── repositories/
│   └── user_repo.py           # DB queries (Module 5)
└── db/
    ├── session.py             # Async engine + session factory (Module 5)
    └── base.py                # Base model class
```

## Key rules
1. **Never import from models in endpoints directly** — always go through services
2. **Never expose ORM models in responses** — always use Pydantic schemas
3. **All config lives in config.py** — no os.environ anywhere else
4. **Lifespan manages all resource lifecycle** — no global startup code

## Install
```bash
pip install fastapi uvicorn pydantic-settings python-dotenv
```

## .env example
```
ENVIRONMENT=development
SECRET_KEY=your-256-bit-secret-key
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379/0
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
```

## Run
```bash
uvicorn main:app --reload --port 8000
```
