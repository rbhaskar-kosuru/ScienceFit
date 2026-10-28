"""Pipeline tests. No Ollama or reranker download needed: both are faked."""
import math

import pytest
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
              lambda system, prompt: "Ten or more weekly sets [schoenfeld_2017]. See also [aagaard_2003].")
    res = rag.ask("How many sets for hypertrophy?")
    assert not res["refused"]
    assert [s["paper_id"] for s in res["sources"]] == ["schoenfeld_2017"]  # helms_2018 fell below threshold
    assert res["sources"][0]["cited"] and res["sources"][0]["title"] == "Volume"
    assert res["unverified_citations"] == ["aagaard_2003"]


def test_quality_score_breaks_relevance_ties():
    rag = RAG(FakeStore([hit("weak_2010", score=0.3), hit("strong_2019", score=0.9)]),
              FakeReranker([0.8, 0.8]), lambda s, p: "x")
    assert [h["paper"] for h in rag.rerank("q", rag.store.query("q"))] == ["strong_2019", "weak_2010"]


def test_validate_citations_handles_groups_prefixes_and_links():
    cited, invented = validate_citations("A [x_2020, y_2021] B [paper_id: zed_2019] C [link](https://u.org)",
                                         {"x_2020", "y_2021"})
    assert cited == {"x_2020", "y_2021"}
    assert invented == ["zed_2019"]


def test_validate_citations_flags_author_style_and_ignores_statistics():
    answer = ("LBM rose [P < 0.05] [n = 12] [95% CI 1.1–2.3, 2019 cohort] [Table 2] [cribb_2007] [smith_2019b] "
              "[Cribb et al.] [Smith 2019] [Smith and Jones, 2018] [cribb_2007; Lee 2020]")
    cited, invented = validate_citations(answer, {"cribb_2007"})
    assert cited == {"cribb_2007"}
    assert invented == ["smith_2019b", "Cribb et al.", "Smith 2019", "Smith and Jones, 2018", "Lee 2020"]


def test_citation_keys_are_readable_unique_and_stable():
    ids = curate.IdAssigner({})
    assert ids.assign(paper("Brad J. Schoenfeld", "doi1")) == "schoenfeld_2017"
    assert ids.assign(paper("Brad Schoenfeld", "doi2")) == "schoenfeld_2017b"
    assert ids.assign(paper("Brad J. Schoenfeld", "doi1")) == "schoenfeld_2017"
    assert curate.citation_key("José Núñez", 2020, "") == "nunez_2020"


def test_loader_ingests_only_included_papers_and_warns_about_the_rest(tmp_path, monkeypatch):
    for name in ("schoenfeld_2017.pdf", "weak_2010.pdf", "my_notes.pdf"):
        (tmp_path / name).touch()
    monkeypatch.setattr(PaperLoader, "_extract", staticmethod(lambda path: "body"))
    records = [{"paper_id": "schoenfeld_2017", "pdf_file": "schoenfeld_2017.pdf", "included": True,
                "title": "Volume", "year": 2017, "doi": None, "score": 0.6},
               {"paper_id": "weak_2010", "pdf_file": "weak_2010.pdf", "included": False, "reason": "score 0.4 below 0.65"},
               {"paper_id": "gone_2015", "pdf_file": "gone_2015.pdf", "included": True}]
    loader = PaperLoader(tmp_path, records)
    docs = {pid: meta for pid, _, meta in loader.load()}
    assert docs == {"schoenfeld_2017": {"title": "Volume", "year": 2017, "score": 0.6, "curated": True}}
    assert loader.warnings() == ["  excluded weak_2010: score 0.4 below 0.65", "  not in manifest: my_notes.pdf"]


def test_vector_store_replaces_stale_chunks_and_returns_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "embed", lambda texts: [[float(len(t)), 1.0, 0.5] for t in texts])
    store = VectorStore(path=tmp_path)
    store.add("a_2020", ["one", "two", "three"], {"title": "A", "curated": True})
    store.add("a_2020", ["only chunk"], {"title": "A", "curated": True})
    assert store.count() == 1
    top = store.query("question")[0]
    assert (top["paper"], top["title"], top["chunk_index"], top["text"]) == ("a_2020", "A", 0, "only chunk")


def test_manifest_save_replaces_file_whole_and_splits_included(tmp_path):
    path = tmp_path / "m.json"
    manifest.save([{"paper_id": "a", "included": True}, {"paper_id": "b", "included": False}], path)
    records = manifest.load(path)
    assert [r["paper_id"] for r in manifest.included(records)] == ["a"]
    assert [r["paper_id"] for r in manifest.excluded(records)] == ["b"]
    assert [f.name for f in tmp_path.iterdir()] == ["m.json"]  # temp file was swapped in, not left behind


def test_split_sections_drops_front_matter_and_references_but_keeps_trailing_tables():
    body = "Methods. Ten sets per week.\n" * 20
    text = ("Downloaded from repository\nAuthor affiliations\nAbstract\nWe studied volume.\n" + body +
            "References\n1. Smith J. Volume. 2017.\n2. Lee K. Load. 2019.\nTable 1. Participant characteristics\n")
    front, kept, refs = ingest.split_sections(text)
    assert front == "Downloaded from repository\nAuthor affiliations\n"
    assert kept.startswith("Abstract") and "Table 1. Participant characteristics" in kept and "Smith J" not in kept
    assert refs.startswith("References\n1. Smith J")


def test_split_sections_keeps_unlabelled_abstract_and_ignores_early_reference_mentions():
    text = "Title\nWe found larger gains.\nINTRODUCTION\nreferences to earlier work\n" + "Body text.\n" * 50
    front, kept, refs = ingest.split_sections(text)
    assert (front, refs) == ("", "") and kept == text


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


def test_scorer_uses_fixed_log_citation_scale():
    def scored(cpy):
        return curate.Paper(**{**vars(paper("A", f"d{cpy}")), "citations_per_year": cpy})
    top, mid, none = curate.Scorer().score_all([scored(0), scored(500), scored(10)])
    assert top.score == round(0.4 + 0.35 * 0.8 + 0.25 * 0.5, 4)  # clipped at CITATION_CAP
    assert mid.score == round(0.4 * math.log1p(10) / math.log1p(50) + 0.28 + 0.125, 4)
    assert none.score == 0.405


def candidate(doi, score, text="resistance training", **extra):
    return curate.Paper(**{**vars(paper("A", doi)), "score": score, "text": text,
                           "pdf_urls": [f"https://h.org/{doi}"], **extra})


def test_judge_includes_every_qualifying_pdf_and_downloads_at_most_target_new_ones(tmp_path):
    for name in ("kept_2017", "late_2016"):
        (tmp_path / f"{name}.pdf").write_bytes(b"%PDF-1")
    ranked = [
        candidate("kept", 0.9, paper_id="kept_2017"),                 # already on disk: no download
        candidate("blocked", 0.85),                                   # download fails
        candidate("good", 0.8),                                       # downloaded: target of 1 met
        candidate("offtopic", 0.75, text="endurance training"),       # gate
        candidate("pulled", 0.72, retracted=True),
        candidate("extra", 0.7),                                      # would need a 2nd download
        candidate("late", 0.68, paper_id="late_2016"),                # on disk: included, no library cap
        candidate("weak", 0.5, paper_id="weak_2012"),                 # below cutoff; old manifest paper
    ]
    tried = []
    def fetch(p):
        tried.append(p.doi)
        return "HTTP 403" if p.doi == "blocked" else None
    attempts = curate.judge(ranked, target=1, cutoff=0.65, fetch=fetch, assign_id=lambda p: p.doi + "_id",
                            papers_dir=tmp_path)
    assert attempts == 2 and tried == ["blocked", "good"]
    assert {p.doi: (p.included, p.reason) for p in ranked} == {
        "kept": (True, None), "blocked": (False, "no valid PDF (HTTP 403)"), "good": (True, None),
        "offtopic": (False, "failed relevance gate"), "pulled": (False, "retracted"),
        "extra": (False, "download target of 1 reached"), "late": (True, None),
        "weak": (False, "score 0.500 below 0.65")}
    assert [p.paper_id for p in ranked if p.paper_id] == ["kept_2017", "blocked_id", "good_id", "late_2016", "weak_2012"]


def test_judge_without_fetch_downloads_nothing(tmp_path):
    ranked = [candidate("new", 0.9)]
    assert curate.judge(ranked, 5, 0.65, fetch=None, assign_id=lambda p: "x", papers_dir=tmp_path) == 0
    assert ranked[0].reason == "no PDF on disk (not downloaded)" and ranked[0].paper_id == ""


def test_id_assigner_avoids_names_of_pdfs_already_on_disk():
    ids = curate.IdAssigner({}, taken={"schoenfeld_2017"})
    assert ids.assign(paper("Brad Schoenfeld", "new")) == "schoenfeld_2017b"


def test_download_falls_back_to_next_link(tmp_path):
    session = FakeSession(lambda url, params: FakeResponse(content=b"%PDF-1" if "mirror" in url else b"<html>"))
    p = curate.Paper(**{**vars(paper("A", "d")), "paper_id": "a_2017",
                        "pdf_urls": ["https://pub.org/d", "https://mirror.org/d"]})
    assert curate.PDFDownloader(tmp_path, session=session, host_delay=0).fetch(p) is None
    assert p.pdf_file == "a_2017.pdf" and len(session.calls) == 2


class FakeClient:
    def __init__(self, known=(), fresh=(), fail=False):
        self.known, self.fresh, self.fail, self.asked = list(known), list(fresh), fail, []

    def fetch_known(self, records):
        self.asked = [r["doi"] for r in records]
        return self.known

    def search_many(self, queries, total):
        if self.fail:
            raise RuntimeError("OpenAlex down")
        return self.fresh


def test_curate_rejudges_manifest_papers_with_fresh_data_and_never_deletes_pdfs(tmp_path, monkeypatch):
    monkeypatch.setattr(curate.config, "CURATE_MIN_SCORE", 0.6)
    papers_dir, path = tmp_path / "papers", tmp_path / "m.json"
    papers_dir.mkdir()
    (papers_dir / "old_2010.pdf").write_bytes(b"%PDF-old")
    manifest.save([{"paper_id": "old_2010", "doi": "old", "pdf_file": "old_2010.pdf", "included": True, "score": 0.9},
                   {"paper_id": "lost_2011", "doi": "lost", "pdf_file": "lost_2011.pdf", "included": True}], path)
    # Re-fetched: same paper, but now off-topic, so it drops out. "lost" isn't found in OpenAlex.
    old_now = raw_work("old", title="Endurance running economy", publication_year=2010, cited_by_count=0)
    client = FakeClient(known=[old_now])
    curate.curate(["q"], target=5, download=False, save=False, client=client, papers_dir=papers_dir,
                  manifest_path=path)
    assert client.asked == ["old", "lost"]
    assert manifest.load(path)[0]["included"] is True  # dry run: manifest untouched

    curate.curate(["q"], target=5, client=client, papers_dir=papers_dir, manifest_path=path)
    records = {r["paper_id"]: r for r in manifest.load(path)}
    assert (records["old_2010"]["included"], records["old_2010"]["reason"]) == (False, "failed relevance gate")
    assert (records["lost_2011"]["included"], records["lost_2011"]["reason"]) == (False, "not found in OpenAlex")
    assert (papers_dir / "old_2010.pdf").exists()


def test_rejudge_refetches_manifest_papers_without_searching_or_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(curate.config, "CURATE_MIN_SCORE", 0.6)
    papers_dir, path = tmp_path / "papers", tmp_path / "m.json"
    papers_dir.mkdir()
    for name in ("strong_2019", "faded_2018"):
        (papers_dir / f"{name}.pdf").write_bytes(b"%PDF-1")
    manifest.save([
        {"paper_id": "strong_2019", "doi": "strong", "pdf_file": "strong_2019.pdf", "included": False, "reason": "outside top 20"},
        {"paper_id": "faded_2018", "doi": "faded", "pdf_file": "faded_2018.pdf", "included": True},
        {"paper_id": "nopdf_2017", "doi": "nopdf", "pdf_file": None, "included": False, "reason": "no valid PDF (HTTP 403)"},
    ], path)
    meta = {"title": "Resistance training meta-analysis", "publication_year": 2019,
            "primary_location": {"source": {"display_name": "Sports Medicine"}}}
    client = FakeClient(known=[raw_work("strong", cited_by_count=400, **meta),
                               raw_work("faded", title="Endurance running economy", publication_year=2018),
                               raw_work("nopdf", cited_by_count=400, **meta)])
    client.search_many = lambda *a, **k: pytest.fail("rejudge must not search")
    downloads = []
    monkeypatch.setattr(curate.PDFDownloader, "fetch", lambda self, p: downloads.append(p) or None)

    curate.curate(["q"], download=False, search=False, client=client, papers_dir=papers_dir, manifest_path=path)
    records = {r["paper_id"]: (r["included"], r["reason"]) for r in manifest.load(path)}
    assert records == {"strong_2019": (True, None), "faded_2018": (False, "failed relevance gate"),
                       "nopdf_2017": (False, "no PDF on disk (not downloaded)")}
    assert downloads == [] and sorted(client.asked) == ["faded", "nopdf", "strong"]


def test_cli_flags_map_to_search_download_and_save(monkeypatch):
    calls = []
    monkeypatch.setattr(curate, "curate", lambda queries, **kw: calls.append(kw) or [])
    for argv in (["curate"], ["curate", "--rejudge"], ["curate", "--no-download"], ["curate", "--rejudge", "--no-download"]):
        monkeypatch.setattr("sys.argv", argv)
        curate.main()
    assert [(c["search"], c["download"], c["save"]) for c in calls] == [
        (True, True, True), (False, False, True), (True, False, False), (False, False, False)]


def test_curate_failure_keeps_previous_manifest(tmp_path, monkeypatch, capsys):
    path = tmp_path / "m.json"
    manifest.save([{"paper_id": "a_2020", "doi": "a", "included": True}], path)
    before = path.read_text()
    with pytest.raises(RuntimeError):
        curate.curate(["q"], client=FakeClient(fail=True), papers_dir=tmp_path, manifest_path=path)
    assert path.read_text() == before

    def boom(*args, **kwargs):
        raise RuntimeError("OpenAlex down")
    monkeypatch.setattr(curate, "curate", boom)
    monkeypatch.setattr("sys.argv", ["curate"])
    with pytest.raises(SystemExit) as exit_info:
        curate.main()
    assert exit_info.value.code == 1
    assert "Previous manifest kept" in capsys.readouterr().err


def test_ingest_rebuilds_from_scratch_and_swaps_in_only_when_done(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "embed", lambda texts: [[float(len(t)), 1.0, 0.5] for t in texts])
    monkeypatch.setattr(PaperLoader, "_extract", staticmethod(lambda path: "Abstract\nnew body"))
    VectorStore(path=tmp_path).add("stale_2001", ["old chunk"], {"title": "Stale"})
    (tmp_path / "new_2020.pdf").touch()
    records = [{"paper_id": "new_2020", "pdf_file": "new_2020.pdf", "included": True, "title": "New"}]
    assert ingest.ingest_all(papers_dir=tmp_path, records=records, path=tmp_path) == 1
    store = VectorStore(path=tmp_path)
    assert store.count() == 1 and store.query("q")[0]["paper"] == "new_2020"
    assert [c.name for c in store.client.list_collections()] == [ingest.config.COLLECTION]
