import ollama
from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import config
from .rag import RAG

app = FastAPI(title="ScienceFit")
rag = RAG()


class Query(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(config.TOP_K, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    try:
        ollama.list()
        ollama_up = True
    except Exception:  # connection errors vary by ollama client version
        ollama_up = False
    chunks = rag.store.count()
    return {
        "status": "ok" if ollama_up and chunks else "degraded",
        "ollama": ollama_up,
        "chunks": chunks,
        "llm": config.LLM_MODEL,
        "reranker": config.RERANK_MODEL,
    }


@app.post("/ask")
def ask(q: Query) -> dict:
    return rag.ask(q.question, top_k=q.top_k)
