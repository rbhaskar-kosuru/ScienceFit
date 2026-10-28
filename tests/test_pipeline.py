"""Pipeline tests. No Ollama or reranker download needed: both are faked."""
from src import curate, ingest, manifest
from src.ingest import Chunker, PaperLoader, VectorStore
from src.rag import RAG, NO_EVIDENCE, validate_citations


class FakeStore:
    def __init__(self, hits):
        self.hits = hits

    def query(self, question, k=20):
        return [dict(h) for h in self.hits]


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores

    def score(self, question, passages):
        return self.scores[:len(passages)]


def hit(paper, i=0, **meta):
    return {"text": f"excerpt from {paper}", "paper": paper, "chunk_index": i, **meta}


def paper(author, doi, year=2017):
    return curate.Paper(openalex_id=doi, doi=doi, title="T", first_author=author, year=year, journal="",
                        citations=0, citations_per_year=0, design="rct", is_oa=False, pdf_url=None, retracted=False)


def test_refuses_without_calling_llm_when_nothing_is_relevant():
    calls = []
    rag = RAG(FakeStore([hit("a_2020"), hit("b_2021")]), FakeReranker([0.01, 0.02]),
              lambda system, prompt: calls.append(prompt) or "unused")
    res = rag.ask("Hi")
    assert res == {"answer": NO_EVIDENCE, "refused": True, "sources": [], "unverified_citations": []}
    assert calls == []


def test_answer_keeps_relevant_sources_and_flags_invented_citations():
    rag = RAG(FakeStore([hit("schoenfeld_2017", title="Volume", year=2017), hit("helms_2018")]),
              FakeReranker([0.9, 0.05]),
              lambda system, prompt: "Ten or more weekly sets [schoenfeld_2017]. See also [Aagaard et al.].")
    res = rag.ask("How many sets for hypertrophy?")
    assert not res["refused"]
    assert [s["paper_id"] for s in res["sources"]] == ["schoenfeld_2017"]  # helms_2018 fell below threshold
    assert res["sources"][0]["cited"] and res["sources"][0]["title"] == "Volume"
    assert res["unverified_citations"] == ["Aagaard et al."]


def test_quality_score_breaks_relevance_ties():
    rag = RAG(FakeStore([hit("weak_2010", score=0.3), hit("strong_2019", score=0.9)]),
              FakeReranker([0.8, 0.8]), lambda s, p: "x")
    assert [h["paper"] for h in rag.rerank("q", rag.store.query("q"))] == ["strong_2019", "weak_2010"]


def test_validate_citations_handles_groups_prefixes_and_links():
    cited, invented = validate_citations("A [x_2020, y_2021] B [paper_id: z] C [link](https://u.org)",
                                         {"x_2020", "y_2021"})
    assert cited == {"x_2020", "y_2021"}
    assert invented == ["z"]


def test_citation_keys_are_readable_unique_and_stable():
    ids = curate.IdAssigner({})
    assert ids.assign(paper("Brad J. Schoenfeld", "doi1")) == "schoenfeld_2017"
    assert ids.assign(paper("Brad Schoenfeld", "doi2")) == "schoenfeld_2017b"
    assert ids.assign(paper("Brad J. Schoenfeld", "doi1")) == "schoenfeld_2017"
    assert curate.citation_key("José Núñez", 2020, "") == "nunez_2020"


def test_loader_attaches_manifest_metadata(tmp_path, monkeypatch):
    for name in ("schoenfeld_2017.pdf", "my_notes.pdf"):
        (tmp_path / name).touch()
    monkeypatch.setattr(PaperLoader, "_extract", staticmethod(lambda path: "body"))
    records = [{"paper_id": "schoenfeld_2017", "pdf_file": "schoenfeld_2017.pdf",
                "title": "Volume", "year": 2017, "doi": None, "score": 0.6}]
    docs = {pid: meta for pid, _, meta in PaperLoader(tmp_path, records).load()}
    assert docs["schoenfeld_2017"] == {"title": "Volume", "year": 2017, "score": 0.6, "curated": True}
    assert docs["my_notes"] == {"title": "my_notes", "curated": False}


def test_vector_store_replaces_stale_chunks_and_returns_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "embed", lambda texts: [[float(len(t)), 1.0, 0.5] for t in texts])
    store = VectorStore(path=tmp_path)
    store.add("a_2020", ["one", "two", "three"], {"title": "A", "curated": True})
    store.add("a_2020", ["only chunk"], {"title": "A", "curated": True})
    assert store.count() == 1
    top = store.query("question")[0]
    assert (top["paper"], top["title"], top["chunk_index"], top["text"]) == ("a_2020", "A", 0, "only chunk")


def test_manifest_merge_updates_and_keeps_old_records():
    old = [{"paper_id": "a", "score": 0.5}]
    new = [{"paper_id": "b", "score": 0.7}, {"paper_id": "a", "score": 0.6}]
    merged = manifest.merge(old, new)
    assert [(r["paper_id"], r["score"]) for r in merged] == [("b", 0.7), ("a", 0.6)]


def test_chunker_respects_size_and_overlap():
    chunks = Chunker(size=10, overlap=2).split("a" * 25)
    assert len(chunks) == 4 and all(len(c) <= 10 for c in chunks)
