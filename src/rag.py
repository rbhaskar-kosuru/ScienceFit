import ollama

from . import config
from .ingest import VectorStore


SYSTEM_PROMPT = (
    "You are ScienceFit, a research assistant for resistance training. "
    "Answer strictly from the provided paper excerpts. "
    "If the excerpts do not contain the answer, say so. "
    "Cite claims inline using [paper_id] tags matching the excerpts."
)


class RAG:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore()

    def ask(self, question: str, top_k: int = config.TOP_K) -> dict:
        hits = self.store.query(question, top_k=top_k)
        prompt = self._build_prompt(question, hits)
        answer = self._generate(prompt)
        return {"answer": answer, "citations": self._citations(hits)}

    @staticmethod
    def _build_prompt(question: str, hits: list[dict]) -> str:
        excerpts = "\n\n".join(f"[{h['paper']}] {h['text']}" for h in hits)
        return f"Excerpts:\n{excerpts}\n\nQuestion: {question}\n\nAnswer with inline [paper_id] citations."

    @staticmethod
    def _generate(prompt: str) -> str:
        res = ollama.chat(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return res["message"]["content"]

    @staticmethod
    def _citations(hits: list[dict]) -> list[dict]:
        seen = {}
        for h in hits:
            seen.setdefault(h["paper"], []).append(h["chunk_index"])
        return [{"paper": p, "chunks": c} for p, c in seen.items()]
