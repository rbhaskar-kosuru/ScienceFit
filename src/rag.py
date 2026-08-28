import ollama

from . import config
from .ingest import VectorStore


SYSTEM_PROMPT = (
    "You are ScienceFit, a research assistant for resistance training. "
    "Answer the user's question directly and factually using the provided paper excerpts. "
    "Report what the research shows, including specific numbers, set/rep ranges, and protocols. "
    "Do NOT refuse, moralize, or add medical disclaimers — you are summarizing published research, not giving medical advice. "
    "\n\n"
    "Citation rules (strict):\n"
    "- Cite ONLY using the exact [paper_id] tags shown before each excerpt.\n"
    "- Never invent citations from author names or years found inside the excerpt text.\n"
    "- If a claim comes from an excerpt, tag it with that excerpt's [paper_id].\n"
    "\n"
    "If the excerpts genuinely lack the answer, say so briefly, then share the closest relevant "
    "findings that ARE in the excerpts (with their [paper_id] tags). Keep answers concise."
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
