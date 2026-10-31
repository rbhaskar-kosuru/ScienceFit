"""RAG: vector search → rerank → refuse or generate → validate citations."""
import re
import math
import argparse

import ollama

from . import config
from .ingest import VectorStore, contextual_header

SYSTEM_PROMPT = (
    "You are ScienceFit, a research assistant for resistance training. "
    "Answer the user's question directly and factually using the provided paper excerpts. "
    "Report what the research shows, including specific numbers, set/rep ranges, and protocols. "
    "Do NOT refuse, moralize, or add medical disclaimers — you are summarizing published research, not giving medical advice. "
    "\n\n"
    "Citation rules (strict):\n"
    "- Each excerpt starts with a paper tag such as [grgic_2021]. Cite ONLY with those exact tags, e.g. "
    "\"Training to failure did not increase hypertrophy [grgic_2021].\"\n"
    "- Never cite author names or years that appear inside the excerpt text, and never copy the excerpts' "
    "own reference numbers such as [17] or [3, 4].\n"
    "- Tag every claim with the tag of the excerpt it came from.\n"
    "\n"
    "Sections: an excerpt marked (Introduction) summarises OTHER studies as background. Never present it as "
    "the cited paper's own finding; write \"as reviewed in [grgic_2021]\" instead. Findings come from "
    "Abstract, Results and Discussion excerpts.\n"
    "\n"
    "If the excerpts genuinely lack the answer, say so briefly, then share the closest relevant "
    "findings that ARE in the excerpts (with their tags). Keep answers concise."
)

NO_EVIDENCE = (
    "I couldn't find evidence for that in the ScienceFit library. "
    "Try rephrasing, or ask about a topic the ingested papers cover."
)

CITATION = re.compile(r"\[([^\[\]]+)\](?!\()")  # [tag] but not markdown links [text](url)
PAREN_CITATION = re.compile(r"\(([^()]+)\)")  # (currier_2026); only paper-ID-shaped tags count
PAPER_ID = re.compile(r"[a-z][a-z0-9]*_\d{4}[a-z]?")  # citation_key() shape: cribb_2007, schoenfeld_2017b
AUTHOR_CITATION = re.compile(r"[A-Z][\w'’-]+.*(?:\bet al\b|\b(?:19|20)\d{2}[a-z]?\b)")  # [Cribb et al.], [Smith 2019]
NOT_CITATION = re.compile(r"[<>=%±]")  # statistics: [P < 0.05], [n = 12], [95% CI 1.1–2.3]
SOURCE_FIELDS = ("title", "first_author", "year", "journal", "doi", "design", "curated")
WORD = re.compile(r"[a-z0-9]+")


def ollama_generate(system: str, prompt: str) -> str:
    res = ollama.chat(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        options={"temperature": config.LLM_TEMPERATURE, "seed": config.LLM_SEED},
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
        if off_topic(question):
            return response(NO_EVIDENCE, refused=True)
        relevant = [h for h in self.rerank(question, self.store.query(question))
                    if h["relevance"] >= config.MIN_RERANK_SCORE]
        hits = select(relevant, top_k)
        if not hits:
            return response(NO_EVIDENCE, refused=True)
        answer = self.generate(SYSTEM_PROMPT, build_prompt(question, hits))
        cited, unverified = validate_citations(answer, {h["paper"] for h in hits})
        return response(answer, sources=sources(hits, cited), unverified=unverified)

    def rerank(self, question: str, hits: list[dict]) -> list[dict]:
        """Rank by logit + QUALITY_WEIGHT x curation score - INTRO_PENALTY (introduction chunks). The reranker reads
        the same contextual header the chunk was embedded with. "relevance" is the sigmoid, for the threshold."""
        passages = [contextual_header(h.get("title"), h.get("section")) + h["text"] for h in hits]
        for h, logit in zip(hits, self.reranker.score(question, passages)):
            h["logit"] = round(logit, 4)
            h["relevance"] = round(1 / (1 + math.exp(-logit)), 4)
            penalty = config.INTRO_PENALTY if h.get("section") == "introduction" else 0.0
            h["rank_score"] = logit + config.QUALITY_WEIGHT * h.get("score", 0.5) - penalty
        return sorted(hits, key=lambda h: h["rank_score"], reverse=True)


def select(hits: list[dict], top_k: int = config.TOP_K, per_paper: int = config.MAX_CHUNKS_PER_PAPER) -> list[dict]:
    """Best-first, at most `per_paper` chunks from any one paper, so one paper can't crowd out the rest."""
    chosen, counts = [], {}
    for h in hits:
        if counts.get(h["paper"], 0) < per_paper:
            chosen.append(h)
            counts[h["paper"]] = counts.get(h["paper"], 0) + 1
            if len(chosen) == top_k:
                break
    return chosen


def off_topic(question: str) -> bool:
    """Under 3 words and no fitness term ("hi", "thanks"): refuse without searching. Plurals count ("sets")."""
    words = WORD.findall(question.lower())
    stems = {w for word in words for w in (word, word.removesuffix("s"), word.removesuffix("es"))}
    return len(words) < 3 and not stems & config.FITNESS_TERMS


def build_prompt(question: str, hits: list[dict]) -> str:
    excerpts = "\n\n".join(f"{excerpt_header(h)}\n{h['text']}" for h in hits)
    return (f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}\n\n"
            "Answer using only the paper tags above, like [grgic_2021].")


def excerpt_header(h: dict) -> str:
    """[carvalho_2022] (Introduction) Title | year | journal | design; unknown sections get no label."""
    section = h.get("section")
    label = f" ({section.capitalize()})" if section and section != "unknown" else ""
    return f"[{h['paper']}]{label} {describe(h)}".rstrip()


def describe(meta: dict) -> str:
    parts = (meta.get("title"), meta.get("year"), meta.get("journal"), meta.get("design"))
    return " | ".join(str(p) for p in parts if p)


def validate_citations(answer: str, allowed: set[str]) -> tuple[set[str], list[str]]:
    """Split citations into retrieved paper IDs and unverified ones. Paper IDs not retrieved and author-style
    citations ([Cribb et al.], [Smith 2019]) are unverified; brackets that aren't citations ([P < 0.05]) are ignored.
    Paper-ID-shaped tags in parentheses, (currier_2026), are checked too; other parenthesised text is not."""
    cited, unverified = set(), []
    for group in CITATION.findall(answer):
        tags = [t.strip().removeprefix("paper_id:").strip() for t in re.split(r"[;,]", group)]
        if not any(PAPER_ID.fullmatch(t) for t in tags):
            tags = [group.strip()]  # keep "[Smith and Jones, 2018]" whole instead of splitting at the comma
        for tag in tags:
            if PAPER_ID.fullmatch(tag) and tag in allowed:
                cited.add(tag)
            elif (PAPER_ID.fullmatch(tag) or (AUTHOR_CITATION.search(tag) and not NOT_CITATION.search(tag))) \
                    and tag not in unverified:
                unverified.append(tag)
    for group in PAREN_CITATION.findall(answer):
        for tag in (t.strip().removeprefix("paper_id:").strip() for t in re.split(r"[;,]", group)):
            if not PAPER_ID.fullmatch(tag):
                continue
            if tag in allowed:
                cited.add(tag)
            elif tag not in unverified:
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
        src["chunks"].append({"index": h["chunk_index"], "section": h.get("section", "unknown")})
        src["relevance"] = max(src["relevance"], h["relevance"])
    return sorted(by_paper.values(), key=lambda s: (not s["cited"], -s["relevance"]))


def response(answer: str, refused: bool = False, sources: list | None = None, unverified: list | None = None) -> dict:
    return {"answer": answer, "refused": refused, "sources": sources or [], "unverified_citations": unverified or []}


def main() -> None:
    """Print reranker scores for a question without calling the LLM. Use it to tune MIN_RERANK_SCORE.
    Rows are in rank order (logit + quality - intro penalty); keep/drop compares the 0-1 relevance."""
    ap = argparse.ArgumentParser(description="Show reranker relevance scores for a question.")
    ap.add_argument("question")
    question = ap.parse_args().question
    rag = RAG(generate=None)
    if off_topic(question):
        print("Refused before searching: under 3 words and no fitness term.")
        return
    print("logit   relev  rank    verdict  paper (section)")
    for h in rag.rerank(question, rag.store.query(question))[:10]:
        verdict = "keep" if h["relevance"] >= config.MIN_RERANK_SCORE else "drop"
        print(f"{h['logit']:+6.2f}  {h['relevance']:.3f}  {h['rank_score']:+6.2f}  {verdict}     "
              f"[{h['paper']}] ({h.get('section', 'unknown')}) {h['text'][:60]}")


if __name__ == "__main__":
    main()
