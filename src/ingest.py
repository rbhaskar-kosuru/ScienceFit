"""Ingest: included manifest papers → strip line numbers, front matter and references → label sections
→ chunks (never crossing a section) → embeddings → Chroma.

Always rebuilds from scratch into a staging collection, then swaps it in, so a failed run keeps the old index.
"""
import re
import logging
import argparse
from pathlib import Path
from typing import Iterator, NamedTuple

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


# Body section headings, compared with whitespace removed and lowercased: small-caps and letter-spaced headings
# come out of pypdf as "REsulT s" or "RESUL TS".
HEADING_LABELS = [
    (re.compile(r"abstract|summary"), "abstract"),
    (re.compile(r"introduction|background"), "introduction"),
    (re.compile(r"(?:(?:materials?|patients|subjects|participants)and)?methods?|methodsandmaterials|methodology"), "methods"),
    (re.compile(r"results?(?:anddiscussion)?"), "results"),
    (re.compile(r"discussion(?:andconclusions?)?|practicalapplications|limitations"), "discussion"),
]
CONCLUSION = re.compile(r"(?:summaryand)?conclusions?(?:and[a-z]+)?|concludingremarks")
KEYWORDS = re.compile(r"key[ \t]*words?[ \t]*:.*", re.I)
NUMBERED = re.compile(r"(\d{1,3}(?:\.\d{1,2})*)\.?(?=\D)")
TOPIC_LINE = re.compile(r"^[ \t]*(\d{1,2})\.?[ \t\xa0]+([A-Z][^\d\n]{2,80}?)[ \t]*\f?$", re.M)
AFFILIATION = re.compile(r"\b(?:department|university|school|institute|faculty|cent(?:re|er)|laboratory|college|hospital)\b", re.I)
TABLE_CAPTION = re.compile(r"^[ \t]*(?:\d{1,4}[ \t]+)?(?:Table|TABLE)[ \t\xa0]*\d+"
                           r"(?:[ \t\xa0]*[.:|]?[ \t\xa0]*(?:\(?[Cc]ontinued\)?|[A-Z‘“\"].*))?[ \t]*\f?$", re.M)
MID_SENTENCE = re.compile(r"(?:\(|\b(?:see|in|and|to|of|from|by|refer|per))\s*$", re.I)
ABSTRACT_MAX = 4000       # chars; with no keywords line or Introduction heading, headings count again after this
TABLE_PROSE_LINES = 6     # this many prose lines among 7 in a row end a table
AFTER_REFERENCES_MARK = "\f\f\n"  # split_sections puts it before tables/figures that follow the references
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


class Heading(NamedTuple):
    start: int
    label: str
    name: str           # squashed, lowercase: "materialsandmethods"
    number: str | None  # "2", "4.3" or None


def find_headings(text: str) -> list[Heading]:
    """Lines that are nothing but a section heading, maybe numbered, letter-spaced, or broken over several very
    short lines (small caps)."""
    found, pos, lines = [], 0, []
    for line in text.split("\n"):
        lines.append((pos, "".join(line.split())))
        pos += len(line) + 1
    for i, (start, squashed) in enumerate(lines):
        if heading := _heading(start, squashed):
            found.append(heading)
        elif 0 < len(squashed) <= 5:
            for n in range(2, 7):
                parts = [s for _, s in lines[i:i + n]]
                if any(not 0 < len(s) <= 5 for s in parts):
                    break
                if heading := _heading(start, "".join(parts)):
                    found.append(heading)
                    break
    return found


def _heading(start: int, squashed: str) -> Heading | None:
    if not squashed or len(squashed) > 45:
        return None
    m = NUMBERED.match(squashed)
    title = squashed[m.end():] if m else squashed
    if not title[:1].isupper():  # a wrapped sentence, not a heading
        return None
    name = title.rstrip(":.").lower()
    label = next((label for pattern, label in HEADING_LABELS if pattern.fullmatch(name)),
                 "discussion" if CONCLUSION.fullmatch(name) else None)
    return Heading(start, label, name, m.group(1) if m else None) if label else None


def body_headings(text: str) -> list[Heading]:
    """The headings that really start a section: one per label, in reading order, plus numbered topic sections."""
    headings = find_headings(text)
    numbered = [h for h in headings if h.number and h.number.isdigit() and int(h.number) <= 15]
    if len(numbered) < 2 or any(int(a.number) >= int(b.number) for a, b in zip(numbered, numbered[1:])):
        numbered = []  # not a numbered paper (or stray manuscript line numbers)
    abstract = next((h for h in headings if h.label == "abstract" and h.start < len(text) / 4), None)
    end = _abstract_end(text, abstract, headings, numbered) if abstract else 0
    by_number = {h.label for h in numbered}

    kept, seen = [], set()
    for i, h in enumerate(headings):
        if h.label == "abstract" or h.start < end or h.label in seen:
            continue
        if h.label in by_number and h not in numbered:
            continue  # e.g. a "Conclusion" table column in a paper whose real one is "5 Conclusion"
        if CONCLUSION.fullmatch(h.name) and any(x.label == "results" or x.name.startswith("discussion") for x in headings[i + 1:]):
            continue  # conclusions come last
        kept.append(h)
        seen.add(h.label)
    if abstract:
        kept.insert(0, abstract)
        if not any(h.start == end for h in kept) and end < len(text) and "introduction" not in seen:
            kept.append(Heading(end, "introduction", "", None))  # abstract over, but no Introduction heading
    if numbered:
        kept += _topic_headings(text, numbered, headings)
    return sorted(kept)


def _abstract_end(text: str, abstract: Heading, headings: list[Heading], numbered: list[Heading]) -> int:
    """Where an abstract stops. Its own "Methods"/"Results" lines (a structured abstract) don't end it; a numbered
    heading, an Introduction heading or the keywords line does. Otherwise the first heading after ABSTRACT_MAX."""
    later = [h for h in headings if h.start > abstract.start]
    ends = [h.start for h in later if h in numbered or h.name == "introduction"][:1]
    limit = min(ends, default=len(text))
    keywords = KEYWORDS.search(text, abstract.start, min(limit, abstract.start + 2 * ABSTRACT_MAX))
    if keywords:
        after = next((h.start for h in later if h.label == "introduction" and 0 <= h.start - keywords.end() < 2500), None)
        ends.append(after or keywords.end())
    return min(ends) if ends else next((h.start for h in later if h.start > abstract.start + ABSTRACT_MAX), len(text))


def _topic_headings(text: str, numbered: list[Heading], headings: list[Heading]) -> list[Heading]:
    """Narrative reviews have topic sections ("2 Training Frequency") between "1 Introduction" and the conclusion.
    Each ends the introduction and counts as discussion. Ignored while a Methods/Results heading is still to come."""
    taken = {int(h.number): h.start for h in numbered}
    last_special = max((h.start for h in headings if h in numbered and h.label in ("methods", "results")), default=-1)
    lines = [(m.start(), int(m.group(1)), m.group(2)) for m in TOPIC_LINE.finditer(text)]
    topics, k, pos = [], min(taken), taken[min(taken)]
    while True:
        k += 1
        if k in taken:
            pos = taken[k]
            continue
        stop = min((p for n, p in taken.items() if n > k), default=len(text))
        match = next((start for j, (start, n, title) in enumerate(lines)
                      if n == k and pos < start < stop and not AFFILIATION.search(title) and not title.rstrip().endswith((",", ".", ";"))
                      and not any(n2 == k + 1 and start2 - start < 300 for start2, n2, _ in lines[j + 1:j + 2])), None)
        if match is None:
            return topics
        pos = match
        if match > last_special:
            topics.append(Heading(match, "discussion", "", str(k)))


def is_prose(line: str) -> bool:
    """A line of running text rather than a table row: long, mostly letters, several lowercase words."""
    s = line.strip()
    letters = sum(c.isalpha() for c in s)
    return len(s) >= 38 and letters >= 0.8 * len(s.replace(" ", "")) and len(re.findall(r"\b[a-z]{2,}\b", s)) >= 5


def table_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of tables: from a "Table N" caption to the end of the page, or until running text resumes.
    Everything after the references marker counts too. Needs the form feeds _extract puts between pages."""
    mark = text.find(AFTER_REFERENCES_MARK)
    body_end = mark if mark >= 0 else len(text)
    spans = []
    for m in TABLE_CAPTION.finditer(text, 0, body_end):
        page_start = text.rfind("\f", 0, m.start()) + 1
        page_end = text.find("\f", m.start(), body_end)
        page_end = body_end if page_end < 0 else page_end + 1
        before, rest = text[page_start:m.start()], text[m.end():page_end].split("\n")
        if re.fullmatch(r"\s*table[ \t\xa0]*\d+\.\s*", m.group(), re.I) or MID_SENTENCE.search(before) \
                or sum(map(is_prose, rest[1:9])) >= 4:
            continue  # the text mentions a table at a line break
        start = page_start if len(before) < 400 and sum(map(is_prose, before.split("\n"))) < 3 else m.start()
        end, pos, recent = page_end, m.end(), []
        for line in rest:
            recent = [*recent, (pos, is_prose(line))][-TABLE_PROSE_LINES - 1:]
            if sum(prose for _, prose in recent) == TABLE_PROSE_LINES:
                end = next(p for p, prose in recent if prose)
                break
            pos += len(line) + 1
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(end, spans[-1][1]))
        else:
            spans.append((start, end))
    return spans + ([(mark, len(text))] if mark >= 0 else [])


def label_sections(text: str) -> list[tuple[str, str]]:
    """[(label, text)] in reading order. Labels: abstract, introduction, methods, results, discussion, table
    (tables, and figures/appendices after the references), and "unknown" before the first heading."""
    spans = table_spans(text)
    marks = [(0, "unknown"), *((h.start, h.label) for h in body_headings(text)
                               if not any(a <= h.start < b for a, b in spans))]
    cuts = sorted({0, len(text), *(p for p, _ in marks), *(p for span in spans for p in span)})
    out: list[tuple[str, str]] = []
    for a, b in zip(cuts, cuts[1:]):
        label = "table" if any(s <= a < e for s, e in spans) else [lab for p, lab in marks if p <= a][-1]
        piece = text[a:b]
        if out and (out[-1][0] == label or len(piece.strip()) < 60 and label == "table"):
            out[-1] = (out[-1][0], out[-1][1] + piece)
        elif piece.strip():
            out.append((label, piece))
    return out


# Second-hand: text that reports other people's studies. Citation = "Smith et al.", "(Smith, 2019)", "[12, 13]",
# "(12–14)" or a superscript number stuck to a word ("strategies12–14").
NUMBERS = r"\d{1,3}(?:\s*[,;–-]\s*\d{1,3})*"
CITED_STUDY = re.compile(rf"\bet\s+al\b|\([^()]*\b(?:19|20)\d\d[a-z]?\b[^()]*\)|\[\s*{NUMBERS}\s*\]|\(\s*{NUMBERS}\s*\)"
                         rf"|[a-z]{{4,}}[.,]?{NUMBERS}(?![\d.%])")
OWN_WORK = re.compile(r"\b(?:we|our|(?:the|this) (?:present|current) (?:study|review|meta|analysis|investigation|work|results|findings)"
                      r"|this (?:study|review|meta|investigation|analysis))\b", re.I)
SENTENCE_END = re.compile(r"(?<!\bal)(?<!\bvs)(?<!\bFig)(?<!\bi\.e)(?<!\be\.g)[.!?]\s+(?=[A-Z])")
SECONDHAND_SHARE = 0.5


def is_secondhand(text: str, section: str = "unknown") -> bool:
    """True if sentences that cite other studies, and don't talk about the paper's own work, make up at least
    SECONDHAND_SHARE of the chunk's characters. Never set for tables (their label already sets them apart) or
    methods (which cite tools and protocols, not findings)."""
    if section in ("table", "methods"):
        return False
    sentences = [s for s in SENTENCE_END.split(text) if s.strip()]
    cited = sum(len(s) for s in sentences if CITED_STUDY.search(s) and not OWN_WORK.search(s))
    return cited >= SECONDHAND_SHARE * max(sum(map(len, sentences)), 1)


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
    after_text = AFTER_REFERENCES_MARK + text[end:] if end < n else ""
    return text[:start], text[start:refs.start()] + after_text, text[refs.start():end]


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
        return "\f\n".join((page.extract_text() or "") for page in reader.pages)  # form feeds mark page ends for tables


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
        narrative = meta.get("design") == "narrative_review"  # no findings of its own: every chunk is second-hand
        self.collection.add(
            ids=[f"{paper_id}::{i}" for i in range(len(chunks))],
            embeddings=embed([config.DOC_PREFIX + contextual_header(meta.get("title"), sec) + c
                              for c, sec in zip(chunks, sections)]),
            documents=chunks,  # raw text: the header is only for embedding
            metadatas=[{**meta, "paper": paper_id, "chunk_index": i, "section": sec,
                        "secondhand": narrative or is_secondhand(c, sec)}
                       for i, (c, sec) in enumerate(zip(chunks, sections))],
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
