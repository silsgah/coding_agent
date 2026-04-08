"""
FastAPI application — the HTTP interface for the coding agent.

Features:
  - Lifespan-managed model loading (no eager import-time GPU allocation)
  - /health endpoint for monitoring
  - /v1/suggest_code — full generation with lint + test results
  - /v1/suggest_code/stream — streaming SSE response
  - /v1/agent/ask — interactive agent loop with tools + memory (Raschka)
  - /v1/agent/session — retrieve session transcript
  - /v1/agent/memory — get current working memory
  - CORS middleware
  - Input validation
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import get_settings
from application.agent import CodingAgent
from domain.session import Session
from infrastructure.session_store import SessionStore
from infrastructure.workspace import WorkspaceContext

logger = logging.getLogger(__name__)

# ── Settings ──────────────────────────────────────────────
settings = get_settings()


# ── Model factory ─────────────────────────────────────────

def _create_llm():
    """Instantiate the correct LLM backend based on config."""
    backend = settings.model.backend

    if backend == "vllm":
        from infrastructure.vllm_model import VLLMModel
        return VLLMModel(settings)
    elif backend == "transformers":
        from infrastructure.llm_model import HFModel
        return HFModel(settings)
    else:
        raise ValueError(f"Unknown backend: {backend!r} (use 'transformers' or 'vllm')")


# ── Shared state (populated during lifespan) ──────────────
_state: dict = {}


# ── Lifespan ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, clean up on shutdown."""
    logger.info("Starting up — loading %s backend…", settings.model.backend)

    try:
        llm = _create_llm()
        llm.load()

        # Build workspace context (Raschka Component 1)
        workspace = WorkspaceContext.build()
        workspace_text = workspace.text()

        # Initialize session store (Raschka Component 5)
        session_store = SessionStore(settings.session.session_dir)

        # Create the original agent (for backward-compat endpoints)
        agent = CodingAgent(llm, settings)

        # Create the interactive agent with workspace + session
        session = Session(workspace_root=workspace.repo_root)
        interactive_agent = CodingAgent(
            llm=llm,
            settings=settings,
            session=session,
            workspace_text=workspace_text,
            max_depth=settings.session.max_depth,
        )

        _state["llm"] = llm
        _state["agent"] = agent
        _state["interactive_agent"] = interactive_agent
        _state["session_store"] = session_store
        _state["workspace"] = workspace
        logger.info("Model loaded — server ready")
    except Exception:
        logger.exception("Failed to load model on startup")
        raise

    yield  # ← server runs here

    # Save session on shutdown
    if "interactive_agent" in _state and "session_store" in _state:
        try:
            _state["session_store"].save(_state["interactive_agent"].session)
            logger.info("Session saved on shutdown")
        except Exception:
            logger.warning("Could not save session on shutdown")

    logger.info("Shutting down — releasing resources")
    _state.clear()


# ── App ───────────────────────────────────────────────────

app = FastAPI(
    title="Mini Coding Agent",
    description="A config-driven coding agent backed by fine-tuned LLMs.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response schemas ────────────────────────────

class CodeRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        examples=["Write a Python function to reverse a linked list"],
    )


class AgentRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        examples=["Fix the failing tests in src/utils.py"],
    )


class LintResponse(BaseModel):
    returncode: int
    stdout: str
    stderr: str


class TestResponse(BaseModel):
    returncode: int
    stdout: str
    stderr: str


class CodeResponse(BaseModel):
    code: str
    lint: LintResponse | None = None
    tests: TestResponse | None = None
    error: str | None = None


# ── Existing endpoints (unchanged) ────────────────────────

@app.get("/health", tags=["ops"])
async def health():
    """Liveness / readiness probe."""
    llm = _state.get("llm")
    ready = llm is not None and llm.is_ready()
    return {
        "status": "ok" if ready else "loading",
        "model": settings.model.name,
        "backend": settings.model.backend,
        "model_ready": ready,
    }


@app.post("/v1/suggest_code", response_model=CodeResponse, tags=["agent"])
async def suggest_code(req: CodeRequest):
    """Generate code (full response with optional lint + test results)."""
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    result = await agent.generate_code(req.description)
    return asdict(result)


@app.post("/v1/suggest_code/stream", tags=["agent"])
async def suggest_code_stream(req: CodeRequest):
    """Stream generated code token-by-token (text/event-stream)."""
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    async def event_stream():
        async for chunk in agent.generate_code_stream(req.description):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── New interactive agent endpoints (Raschka) ─────────────

@app.post("/v1/agent/ask", tags=["interactive"])
async def agent_ask(req: AgentRequest):
    """
    Interactive agent loop — send a message, get a response after
    the agent uses tools, reads files, and reasons about the workspace.
    """
    agent = _state.get("interactive_agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    try:
        response = await agent.ask(req.message)

        # Auto-save session after each interaction
        store = _state.get("session_store")
        if store:
            store.save(agent.session)

        return {
            "response": response,
            "session_id": agent.session.id,
            "tool_steps": len([
                e for e in agent.session.history if e.role == "tool"
            ]),
        }
    except Exception as e:
        logger.exception("Agent ask failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/agent/memory", tags=["interactive"])
async def agent_memory():
    """Get the current working memory of the interactive agent."""
    agent = _state.get("interactive_agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    return agent.session.memory.to_dict()


@app.get("/v1/agent/session/{session_id}", tags=["interactive"])
async def agent_session(session_id: str):
    """Retrieve a saved session transcript by ID."""
    store = _state.get("session_store")
    if store is None:
        raise HTTPException(status_code=503, detail="Session store not initialized")

    try:
        session = store.load(session_id)
        return session.to_dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")


@app.get("/v1/agent/sessions", tags=["interactive"])
async def list_sessions():
    """List all saved session IDs."""
    store = _state.get("session_store")
    if store is None:
        raise HTTPException(status_code=503, detail="Session store not initialized")

    return {"sessions": store.list_sessions()}