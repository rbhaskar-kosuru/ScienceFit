"""Pipeline tests. No Ollama or reranker download needed: both are faked."""
import math
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


# --- Curation ---

class FakeResponse:
    def __init__(self, json_data=None, content=b"", status=200):
        self._json, self.content, self.status_code = json_data, content, status
        self.ok, self.headers = status < 400, {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, respond):
        self.respond, self.headers, self.calls = respond, {}, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.respond(url, params)


def raw_work(doi, **extra):
    return {"id": doi, "doi": doi, "title": "T", "open_access": {"is_oa": True},
            "best_oa_location": {"pdf_url": f"https://x.org/{doi}.pdf"}, **extra}


def test_search_paginates_at_200_with_oa_filter_and_stops_when_results_run_out():
    pages = {1: [raw_work(f"d{i}") for i in range(200)], 2: [raw_work("last")]}
    session = FakeSession(lambda url, params: FakeResponse({"results": pages.get(params["page"], [])}))
    results = curate.OpenAlexClient(mailto="", api_key="", session=session).search("q", n=500)
    assert len(results) == 201 and len(session.calls) == 2
    params = session.calls[0][1]
    assert params["per-page"] == 200 and "type:article|review,is_oa:true" in params["filter"]


def test_search_many_splits_total_across_queries_and_dedupe_removes_overlap():
    session = FakeSession(lambda url, params: FakeResponse({"results": [raw_work("shared"), raw_work(params["search"])]}))
    raw = curate.OpenAlexClient(mailto="", api_key="", session=session).search_many(["a", "b"], total=4)
    assert [p["per-page"] for _, p in session.calls] == [2, 2]
    parser = curate.PaperParser(current_year=2026)
    assert [p.doi for p in curate.dedupe([parser.parse(r) for r in raw])] == ["shared", "a", "b"]


def test_parser_collects_all_oa_pdf_links_best_first():
    raw = raw_work("d", locations=[{"is_oa": True, "pdf_url": "https://x.org/d.pdf"},
                                   {"is_oa": True, "pdf_url": "https://mirror.org/d.pdf"},
                                   {"is_oa": False, "pdf_url": "https://paywall.org/d.pdf"}])
    assert curate.PaperParser().parse(raw).pdf_urls == ["https://x.org/d.pdf", "https://mirror.org/d.pdf"]


def test_gate_and_design_match_whole_words_and_skip_negations():
    gate = curate.RelevanceFilter()
    text = lambda t: curate.Paper(**{**vars(paper("A", "d")), "text": t})
    assert gate.apply([text("strength gains")]) and not gate.apply([text("strengthen the case")])
    assert not gate.apply([text("endurance training and cognitive training")])  # 'training' alone isn't enough
    assert not gate.apply([text("strength in knee osteoarthritis")])
    assert not gate.apply([text("resistance training in diabetes")]) and gate.apply([text("non-clinical resistance training")])
    design = curate.PaperParser._design
    assert design("a randomized trial")[0] == "rct"
    assert design("a non-randomized trial")[0] == "other"
    assert design("a non randomised cohort")[0] == "cohort"
    assert design("muscle structure")[0] == "other"  # 'rct' inside a word doesn't count


def test_journal_names_normalize_ampersands_and_punctuation():
    assert curate.normalize_journal("Medicine & Science in Sports & Exercise") in curate.TRUSTED_JOURNALS
    assert curate.normalize_journal("Scandinavian Journal of Medicine & Science in Sports") in curate.TRUSTED_JOURNALS


def test_scorer_uses_fixed_log_citation_scale_and_min_score():
    def scored(cpy, journal=""):
        return curate.Paper(**{**vars(paper("A", f"d{cpy}{journal}")), "citations_per_year": cpy, "journal": journal})
    top, mid, none = curate.Scorer(min_score=None).score_all([scored(500), scored(10), scored(0)])
    assert top.score == round(0.4 + 0.35 * 0.8 + 0.25 * 0.5, 4)  # clipped at CITATION_CAP
    assert mid.score == round(0.4 * math.log1p(10) / math.log1p(50) + 0.28 + 0.125, 4)
    assert none.score == 0.405
    assert [p.citations_per_year for p in curate.Scorer(min_score=0.5).score_all([scored(500), scored(0)])] == [500]


def test_download_until_walks_rank_order_skips_failures_and_stops_at_target(tmp_path):
    bodies = {"good1": b"%PDF-1", "html": b"<html>", "good2": b"%PDF-2", "good3": b"%PDF-3"}
    session = FakeSession(lambda url, params: FakeResponse(content=bodies[url.split("/")[-1]]))
    papers = [curate.Paper(**{**vars(paper("A", name)), "pdf_urls": [f"https://h.org/{name}"]}) for name in bodies]
    got = curate.PDFDownloader(tmp_path, session=session, host_delay=0).download_until(
        papers, target=2, assign_id=lambda p: p.doi)
    assert [p.paper_id for p in got] == ["good1", "good2"]
    assert len(session.calls) == 3  # good3 never tried
    assert sorted(f.name for f in tmp_path.iterdir()) == ["good1.pdf", "good2.pdf"]


def test_download_falls_back_to_next_link(tmp_path):
    session = FakeSession(lambda url, params: FakeResponse(content=b"%PDF-1" if "mirror" in url else b"<html>"))
    p = curate.Paper(**{**vars(paper("A", "d")), "pdf_urls": ["https://pub.org/d", "https://mirror.org/d"]})
    got = curate.PDFDownloader(tmp_path, session=session, host_delay=0).download_until([p], 1, lambda p: "a_2017")
    assert got[0].pdf_file == "a_2017.pdf"
