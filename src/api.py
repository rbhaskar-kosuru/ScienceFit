from fastapi import FastAPI
from pydantic import BaseModel

from .rag import RAG

app = FastAPI(title="ScienceFit")
rag = RAG()


class Query(BaseModel):
    question: str
    top_k: int = 5


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Query) -> dict:
    return rag.ask(q.question, top_k=q.top_k)
