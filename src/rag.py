"""RAG: vector search → rerank → refuse or generate → validate citations."""
import re
import argparse

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
    "- Each excerpt starts with a [paper_id] tag. Cite ONLY with those exact tags.\n"
    "- Never cite author names or years that appear inside the excerpt text.\n"
    "- Tag every claim with the [paper_id] of the excerpt it came from.\n"
    "\n"
    "If the excerpts genuinely lack the answer, say so briefly, then share the closest relevant "
    "findings that ARE in the excerpts (with their [paper_id] tags). Keep answers concise."
)

NO_EVIDENCE = (
    "I couldn't find evidence for that in the ScienceFit library. "
    "Try rephrasing, or ask about a topic the ingested papers cover."
)

CITATION = re.compile(r"\[([^\[\]]+)\](?!\()")  # [tag] but not markdown links [text](url)
SOURCE_FIELDS = ("title", "first_author", "year", "journal", "doi", "design", "curated")


def ollama_generate(system: str, prompt: str) -> str:
    res = ollama.chat(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    )
    return res["message"]["content"]


class RAG:
    def __init__(self, store=None, reranker=None, generate=ollama_generate):
        self.store = store or VectorStore()
        if reranker is None:
            from .rerank import Reranker  # loads torch; kept out of module import
            reranker = Reranker()
        self.reranker = reranker
        self.generate = generate

    def ask(self, question: str, top_k: int = config.TOP_K) -> dict:
        hits = [h for h in self.rerank(question, self.store.query(question))
                if h["relevance"] >= config.MIN_RERANK_SCORE][:top_k]
        if not hits:
            return response(NO_EVIDENCE, refused=True)
        answer = self.generate(SYSTEM_PROMPT, build_prompt(question, hits))
        cited, unverified = validate_citations(answer, {h["paper"] for h in hits})
        return response(answer, sources=sources(hits, cited), unverified=unverified)

    def rerank(self, question: str, hits: list[dict]) -> list[dict]:
        for h, s in zip(hits, self.reranker.score(question, [h["text"] for h in hits])):
            h["relevance"] = round(s, 4)
            h["rank_score"] = s + config.QUALITY_WEIGHT * h.get("score", 0.5)
        return sorted(hits, key=lambda h: h["rank_score"], reverse=True)


def build_prompt(question: str, hits: list[dict]) -> str:
    excerpts = "\n\n".join(f"[{h['paper']}] ({describe(h)})\n{h['text']}" for h in hits)
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}\n\nAnswer using only the [paper_id] tags above."


def describe(meta: dict) -> str:
    parts = (meta.get("title"), meta.get("year"), meta.get("journal"), meta.get("design"))
    return " | ".join(str(p) for p in parts if p)


def validate_citations(answer: str, allowed: set[str]) -> tuple[set[str], list[str]]:
    """Split [tags] in the answer into real retrieved paper IDs and invented ones."""
    cited, unverified = set(), []
    for group in CITATION.findall(answer):
        for tag in re.split(r"[;,]", group):
            tag = tag.strip().removeprefix("paper_id:").strip()
            if tag in allowed:
                cited.add(tag)
            elif tag and tag not in unverified:
                unverified.append(tag)
    return cited, unverified


def sources(hits: list[dict], cited: set[str]) -> list[dict]:
    by_paper = {}
    for h in hits:
        src = by_paper.setdefault(h["paper"], {
            "paper_id": h["paper"],
            **{k: h[k] for k in SOURCE_FIELDS if k in h},
            "cited": h["paper"] in cited,
            "relevance": h["relevance"],
            "chunks": [],
        })
        src["chunks"].append(h["chunk_index"])
        src["relevance"] = max(src["relevance"], h["relevance"])
    return sorted(by_paper.values(), key=lambda s: (not s["cited"], -s["relevance"]))


def response(answer: str, refused: bool = False, sources: list | None = None, unverified: list | None = None) -> dict:
    return {"answer": answer, "refused": refused, "sources": sources or [], "unverified_citations": unverified or []}


def main() -> None:
    """Print reranker scores for a question without calling the LLM. Use it to tune MIN_RERANK_SCORE."""
    ap = argparse.ArgumentParser(description="Show reranker relevance scores for a question.")
    ap.add_argument("question")
    question = ap.parse_args().question
    rag = RAG(generate=None)
    for h in rag.rerank(question, rag.store.query(question))[:10]:
        verdict = "keep" if h["relevance"] >= config.MIN_RERANK_SCORE else "drop"
        print(f"{h['relevance']:.3f}  {verdict}  [{h['paper']}] {h['text'][:70]}")


if __name__ == "__main__":
    main()
