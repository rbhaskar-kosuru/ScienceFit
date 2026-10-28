"""Ingest: PDFs → chunks → embeddings → Chroma, with curation metadata on every chunk."""
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
    """Yields (paper_id, text, metadata). Curated PDFs get manifest metadata; manual PDFs use the filename."""

    def __init__(self, papers_dir: Path = config.PAPERS_DIR, records: list[dict] | None = None):
        self.papers_dir = papers_dir
        self.by_file = manifest.by_file(manifest.load() if records is None else records)

    def load(self) -> Iterator[tuple[str, str, dict]]:
        for pdf in sorted(self.papers_dir.glob("*.pdf")):
            record = self.by_file.get(pdf.name)
            paper_id = record["paper_id"] if record else pdf.stem
            yield paper_id, self._extract(pdf), self._meta(record, pdf)

    @staticmethod
    def _meta(record: dict | None, pdf: Path) -> dict:
        if not record:
            return {"title": pdf.stem, "curated": False}
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
    def __init__(self, path: Path = config.CHROMA_DIR, reset: bool = False):
        self.client = chromadb.PersistentClient(path=str(path))
        if reset:
            self._drop()
        self.collection = self.client.get_or_create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})

    def _drop(self) -> None:
        try:
            self.client.delete_collection(config.COLLECTION)
        except Exception:  # collection doesn't exist yet; error type differs across chromadb versions
            pass

    def add(self, paper_id: str, chunks: list[str], meta: dict) -> None:
        self.collection.delete(where={"paper": paper_id})  # re-ingesting a paper replaces its old chunks
        if not chunks:
            return
        self.collection.add(
            ids=[f"{paper_id}::{i}" for i in range(len(chunks))],
            embeddings=embed([config.DOC_PREFIX + c for c in chunks]),
            documents=chunks,
            metadatas=[{**meta, "paper": paper_id, "chunk_index": i} for i in range(len(chunks))],
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


def ingest_all(reset: bool = False) -> int:
    loader, chunker, store = PaperLoader(), Chunker(), VectorStore(reset=reset)
    total = 0
    for paper_id, text, meta in loader.load():
        chunks = chunker.split(text)
        store.add(paper_id, chunks, meta)
        total += len(chunks)
        print(f"[{paper_id}] {len(chunks)} chunks ({'curated' if meta['curated'] else 'manual'})")
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest PDFs from data/papers into Chroma.")
    ap.add_argument("--reset", action="store_true", help="drop the collection and rebuild from scratch")
    n = ingest_all(reset=ap.parse_args().reset)
    print(f"Done. {n} chunks ingested.")
