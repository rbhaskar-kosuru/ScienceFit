import re
import hashlib
from pathlib import Path
from typing import Iterable

import ollama
import chromadb
from pypdf import PdfReader

from . import config


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
    def __init__(self, papers_dir: Path = config.PAPERS_DIR):
        self.papers_dir = papers_dir

    def load(self) -> Iterable[tuple[str, str]]:
        for pdf in self.papers_dir.glob("*.pdf"):
            yield pdf.stem, self._extract(pdf)

    @staticmethod
    def _extract(path: Path) -> str:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(config.COLLECTION)

    def add(self, paper_id: str, chunks: list[str]) -> None:
        if not chunks:
            return
        embeddings = [self._embed(c) for c in chunks]
        ids = [self._id(paper_id, i, c) for i, c in enumerate(chunks)]
        metadatas = [{"paper": paper_id, "chunk_index": i} for i in range(len(chunks))]
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    def query(self, question: str, top_k: int = config.TOP_K) -> list[dict]:
        q_embed = self._embed(question)
        res = self.collection.query(query_embeddings=[q_embed], n_results=top_k)
        return [
            {"text": doc, "paper": meta["paper"], "chunk_index": meta["chunk_index"]}
            for doc, meta in zip(res["documents"][0], res["metadatas"][0])
        ]

    @staticmethod
    def _embed(text: str) -> list[float]:
        return ollama.embeddings(model=config.EMBED_MODEL, prompt=text)["embedding"]

    @staticmethod
    def _id(paper_id: str, idx: int, chunk: str) -> str:
        h = hashlib.md5(chunk.encode()).hexdigest()[:8]
        return f"{paper_id}::{idx}::{h}"


def ingest_all() -> int:
    loader = PaperLoader()
    chunker = Chunker()
    store = VectorStore()
    total = 0
    for paper_id, text in loader.load():
        chunks = chunker.split(text)
        store.add(paper_id, chunks)
        total += len(chunks)
        print(f"[{paper_id}] {len(chunks)} chunks")
    return total


if __name__ == "__main__":
    n = ingest_all()
    print(f"Done. {n} chunks ingested.")
