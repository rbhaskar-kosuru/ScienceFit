"""Ingest: included manifest papers → strip line numbers, front matter and references → label sections
→ chunks (never crossing a section) → embeddings → Chroma.

Always rebuilds from scratch into a staging collection, then swaps it in, so a failed run keeps the old index.
"""
import re
import logging
import argparse
from pathlib import Path
from typing import Iterator

import ollama
import chromadb
from pypdf import PdfReader

from . import config, manifest

logging.getLogger("pypdf").setLevel(logging.ERROR)  # silences malformed-PDF noise like /MediaBox warnings

META_FIELDS = ("title", "first_author", "year", "journal", "doi", "design", "score", "citations_per_year")

# Section headings as they come out of pypdf: on their own line, maybe after a line or section number.
FRONT_END = re.compile(r"^\s*(?:abstract|summary)\b", re.I | re.M)
REFERENCES = re.compile(r"^\s*(?:\d+\.?\s*)?(?:references|bibliography|literature cited|reference list)\b", re.I | re.M)
# Author manuscripts put tables and figures after the references; those are kept.
AFTER_REFERENCES = re.compile(r"^\s*(?:\d+\s+)?(?:table|figure|fig\.|appendix|supplementary)\s*[a-z]?\d*\b", re.I | re.M)


# Body section headings: the whole line, maybe numbered ("2.", "3.1"). Several headings map to one label.
HEADING = re.compile(r"^[ \t]*(?:\d+(?:\.\d+)*\.?[ \t]+)?(abstract|summary|introduction|background|materials and methods|"
                     r"methods|methodology|results|discussion|conclusions?|practical applications|limitations)"
                     r"[ \t]*:?[ \t]*$", re.I | re.M)
SECTION_LABELS = {
    "abstract": "abstract", "summary": "abstract",
    "introduction": "introduction", "background": "introduction",
    "materials and methods": "methods", "methods": "methods", "methodology": "methods",
    "results": "results",
    "discussion": "discussion", "conclusion": "discussion", "conclusions": "discussion",
    "practical applications": "discussion", "limitations": "discussion",
}
LINE_NUMBER = re.compile(r"^[ \t]*(\d{1,4})(?=[ \t]|$)[ \t]*", re.M)


def strip_line_numbers(text: str) -> str:
    """Author manuscripts number every line ("457 REFERENCES"). Strip them only when most lines start with an
    increasing number, so numeric table rows in ordinary PDFs are left alone."""
    lines = [line for line in text.splitlines() if line.strip()]
    numbers = [int(m.group(1)) for line in lines if (m := LINE_NUMBER.match(line))]
    if len(numbers) < 0.4 * len(lines) or len(numbers) < 2:
        return text
    rising = sum(b > a for a, b in zip(numbers, numbers[1:])) / (len(numbers) - 1)
    return LINE_NUMBER.sub("", text) if rising >= 0.8 else text


def label_sections(text: str) -> list[tuple[str, str]]:
    """[(label, text)] split at body headings; text before the first heading is "unknown". Inside an abstract,
    "Methods"/"Results"/"Conclusions" lines are a structured abstract, so they're ignored until "Introduction"
    (or until 20% into the text)."""
    marks, current = [], None
    for m in HEADING.finditer(text):
        label = SECTION_LABELS[" ".join(m.group(1).lower().split())]
        if current == "abstract" and m.group(1).lower() != "introduction" and m.start() < len(text) * 0.2:
            continue
        if label != current:
            marks.append((m.start(), label))
            current = label
    bounds = [(0, "unknown"), *marks, (len(text), "")]
    return [(label, text[start:end]) for (start, label), (end, _) in zip(bounds, bounds[1:]) if text[start:end].strip()]


def contextual_header(title: str | None, section: str | None) -> str:
    """"Paper: <title>. Section: <section>.\n", prepended to a chunk for embedding and reranking (not stored),
    so a chunk that says "we found..." still carries what paper and section it comes from. Unknown parts are left out."""
    parts = [f"Paper: {title}." if title else "", f"Section: {section}." if section and section != "unknown" else ""]
    header = " ".join(p for p in parts if p)
    return header + "\n" if header else ""


def split_sections(text: str) -> tuple[str, str, str]:
    """(front matter, body, references). Front matter is everything before an Abstract/Summary heading in the
    first quarter; with no such heading (e.g. Frontiers) nothing is cut, so an unlabelled abstract survives.
    References start at the first such heading in the second half and run to the next table/figure/appendix."""
    n = len(text)
    front = FRONT_END.search(text, 0, n // 4)
    start = front.start() if front else 0
    refs = REFERENCES.search(text, max(start, n // 2))
    if not refs:
        return text[:start], text[start:], ""
    after = AFTER_REFERENCES.search(text, refs.end())
    end = after.start() if after else n
    return text[:start], text[start:refs.start()] + text[end:], text[refs.start():end]


class Chunker:
    def __init__(self, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP):
        self.size = size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= self.size:
            return [text] if text else []
        step = self.size - self.overlap
        return [text[i:i + self.size] for i in range(0, len(text), step) if text[i:i + self.size]]


class PaperLoader:
    """Yields (paper_id, body text, metadata) for papers the manifest marks as included."""

    def __init__(self, papers_dir: Path = config.PAPERS_DIR, records: list[dict] | None = None):
        self.papers_dir = papers_dir
        self.records = manifest.load() if records is None else records

    def load(self) -> Iterator[tuple[str, str, dict]]:
        for r in sorted(manifest.included(self.records), key=lambda r: r["paper_id"]):
            pdf = self.papers_dir / (r.get("pdf_file") or "")
            if not pdf.is_file():
                print(f"Warning: {r['paper_id']} is included but {pdf.name or 'its PDF'} is missing; skipped.")
                continue
            yield r["paper_id"], split_sections(strip_line_numbers(self._extract(pdf)))[1], self._meta(r)

    def warnings(self) -> list[str]:
        """Excluded manifest papers and PDFs the manifest doesn't know about. Neither is ingested."""
        lines = [f"  excluded {r['paper_id']}: {r.get('reason') or 'not judged yet (run curate)'}"
                 for r in manifest.excluded(self.records)]
        listed = {r.get("pdf_file") for r in self.records}
        lines += [f"  not in manifest: {pdf.name}" for pdf in sorted(self.papers_dir.glob("*.pdf")) if pdf.name not in listed]
        return lines

    @staticmethod
    def _meta(record: dict) -> dict:
        meta = {k: record[k] for k in META_FIELDS if record.get(k) not in (None, "")}  # Chroma rejects None
        return {**meta, "curated": True}

    @staticmethod
    def _extract(path: Path) -> str:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)


def embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), config.EMBED_BATCH):
        vectors.extend(ollama.embed(model=config.EMBED_MODEL, input=texts[i:i + config.EMBED_BATCH])["embeddings"])
    return vectors


class VectorStore:
    def __init__(self, path: Path = config.CHROMA_DIR, name: str = config.COLLECTION, reset: bool = False):
        self.client = chromadb.PersistentClient(path=str(path))
        self.name = name
        if reset:
            self._drop(name)
        self.collection = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def _drop(self, name: str) -> None:
        try:
            self.client.delete_collection(name)
        except Exception:  # collection doesn't exist yet; error type differs across chromadb versions
            pass

    def promote(self, name: str = config.COLLECTION) -> None:
        """Replace collection `name` with this one (used to swap a finished staging build in)."""
        self._drop(name)
        self.collection.modify(name=name)
        self.name = name

    def add(self, paper_id: str, chunks: list[str], meta: dict, sections: list[str] | None = None) -> None:
        self.collection.delete(where={"paper": paper_id})  # re-ingesting a paper replaces its old chunks
        if not chunks:
            return
        sections = sections or ["unknown"] * len(chunks)
        self.collection.add(
            ids=[f"{paper_id}::{i}" for i in range(len(chunks))],
            embeddings=embed([config.DOC_PREFIX + contextual_header(meta.get("title"), sec) + c
                              for c, sec in zip(chunks, sections)]),
            documents=chunks,  # raw text: the header is only for embedding
            metadatas=[{**meta, "paper": paper_id, "chunk_index": i, "section": sec} for i, sec in enumerate(sections)],
        )

    def query(self, question: str, k: int = config.CANDIDATE_K) -> list[dict]:
        n = min(k, self.count())
        if n == 0:
            return []
        res = self.collection.query(
            query_embeddings=embed([config.QUERY_PREFIX + question]),
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"text": doc, "distance": dist, **meta}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def count(self) -> int:
        return self.collection.count()


def ingest_all(papers_dir: Path = config.PAPERS_DIR, records: list[dict] | None = None,
               path: Path = config.CHROMA_DIR) -> int:
    loader, chunker = PaperLoader(papers_dir, records), Chunker()
    if warnings := loader.warnings():
        print("Warning: these PDFs are NOT ingested:\n" + "\n".join(warnings))
    store = VectorStore(path, name=config.COLLECTION + "_staging", reset=True)
    total = 0
    for paper_id, text, meta in loader.load():
        chunks, sections = [], []
        for label, section_text in label_sections(text):
            pieces = chunker.split(section_text)
            chunks += pieces
            sections += [label] * len(pieces)
        store.add(paper_id, chunks, meta, sections)
        total += len(chunks)
        counts = ", ".join(f"{label} {sections.count(label)}" for label in dict.fromkeys(sections))
        print(f"[{paper_id}] {len(chunks)} chunks ({counts})")
    store.promote()
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rebuild Chroma from the papers the manifest marks as included.")
    ap.add_argument("--reset", action="store_true", help=argparse.SUPPRESS)  # no-op: every run rebuilds
    ap.parse_args()
    n = ingest_all()
    print(f"Done. {n} chunks ingested.")
