"""API FastAPI — RAG-Time Chat
==============================

Lance : uvicorn api:app --reload --port 8000

Endpoints :
  GET  /             → frontend (static/index.html)
  GET  /api/health   → état de l'API et de l'index
  GET  /api/info     → infos sur le vectorstore chargé
  POST /api/chat     → query RAG (retrieval + génération LLM)
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# ── Config par variables d'environnement (ou .env) ──────────────────────────
from dotenv import load_dotenv
load_dotenv()

VECTORSTORE_DIR = ROOT / "data" / "vectorstore_multilingual_test"
EMBED_MODEL     = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
DEFAULT_MODEL   = os.getenv("LLM_MODEL",   "openai/gpt-4o-mini")
TOP_K_DEFAULT   = int(os.getenv("TOP_K", "5"))

# ── App FastAPI ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG-Time API",
    description="API de chat RAG sur tickets de support multilingue",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Instance RAG (chargée au démarrage) ─────────────────────────────────────
_rag = None


def get_rag():
    global _rag
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG engine non initialisé")
    return _rag


@app.on_event("startup")
async def load_rag():
    global _rag
    from rag import RAGEngine
    print(f"[startup] Chargement du vectorstore : {VECTORSTORE_DIR}")
    _rag = RAGEngine(
        embed_model=EMBED_MODEL,
        persist_dir=VECTORSTORE_DIR,
        llm_model=DEFAULT_MODEL,
        top_k=TOP_K_DEFAULT,
    )
    print(f"[startup] Index prêt — {_rag.index_count} chunks | LLM : {DEFAULT_MODEL}")


# ── Schémas Pydantic ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question:    str                = Field(..., min_length=1, max_length=2000)
    model:       str                = Field(default="openai/gpt-4o-mini")
    top_k:       int                = Field(default=5, ge=1, le=20)
    temperature: float              = Field(default=0.3, ge=0.0, le=1.0)
    filters:     dict | None        = Field(default=None)


class Source(BaseModel):
    ticket_id:   str
    similarity:  float
    product:     str
    category:    str
    language:    str
    priority:    str
    excerpt:     str


class ChatResponse(BaseModel):
    answer:      str
    sources:     list[Source]
    num_results: int
    model:       str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    rag = get_rag()
    return {
        "status": "ok",
        "index_count": rag.index_count,
        "has_llm": rag.has_llm,
        "embed_model": EMBED_MODEL,
        "llm_model": rag.llm_model,
    }


@app.get("/api/info")
async def info():
    rag = get_rag()
    return {
        "vectorstore": str(VECTORSTORE_DIR),
        "chunks": rag.index_count,
        "embed_model": EMBED_MODEL,
        "default_llm": rag.llm_model,
        "top_k": TOP_K_DEFAULT,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    rag = get_rag()

    # Changer de modèle LLM à la volée si nécessaire
    if req.model and req.model != rag.llm_model:
        rag.llm_model = req.model

    result = rag.query(
        question=req.question,
        top_k=req.top_k,
        filters=req.filters or None,
        temperature=req.temperature,
    )

    sources = [
        Source(
            ticket_id  = s.get("ticket_id", "?"),
            similarity = round(s.get("similarity", 0.0), 4),
            product    = s.get("product", ""),
            category   = s.get("category", ""),
            language   = s.get("language", ""),
            priority   = s.get("priority", ""),
            excerpt    = s.get("excerpt", ""),
        )
        for s in result.get("sources", [])
    ]

    return ChatResponse(
        answer      = result["answer"],
        sources     = sources,
        num_results = result["num_results"],
        model       = rag.llm_model,
    )


# ── Serveur statique (frontend) ─────────────────────────────────────────────
STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def frontend():
    index_html = STATIC_DIR / "index.html"
    if not index_html.exists():
        return {"message": "Frontend non trouvé — déposez index.html dans static/"}
    return FileResponse(str(index_html))
