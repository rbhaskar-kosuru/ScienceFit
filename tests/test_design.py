"""Study design from titles, PubMed types and keywords. PubMed is a fake session; nothing touches the network."""
import pytest
import requests
from src import curate, ingest, manifest
from src.curate import PubMedClient, declared_design, keyword_design, pubmed_design, set_designs
from tests.test_pipeline import FakeClient, FakeResponse, paper, raw_work


class FakeNCBI:
    """types: {pmid: (publication types, doi)}; dois: {doi: [pmids]}."""
    def __init__(self, types=None, dois=None, status=None):
        self.types, self.dois, self.status, self.headers, self.calls = types or {}, dois or {}, list(status or []), {}, []

    def post(self, url, data=None, timeout=None):
        self.calls.append((url, data))
        if self.status and (code := self.status.pop(0)) != 200:
            return FakeResponse({}, status=code)
        if url.endswith("esearch.fcgi"):
            return FakeResponse({"esearchresult": {"idlist": self.dois.get(data["term"], [])}})
        ids = data["id"].split(",")
        result = {"uids": ids}
        for i in ids:
            types, doi = self.types.get(i, (None, None))
            result[i] = {"error": "cannot get document summary"} if types is None else {
                "pubtype": types, "articleids": [{"idtype": "doi", "value": doi}] if doi else []}
        return FakeResponse({"result": result})


def client(**kw):
    return PubMedClient(email="me@example.com", session=FakeNCBI(**kw), delay=0)


def make(doi, title="T", pmid=None, design="other"):
    return curate.Paper(**{**vars(paper("A", doi)), "title": title, "pmid": pmid, "design": design})


@pytest.mark.parametrize("types, design", [
    (["Journal Article", "Meta-Analysis", "Systematic Review"], "meta_analysis"),
    (["Network Meta-Analysis", "Systematic Review"], "meta_analysis"),
    (["Systematic Review", "Review"], "systematic_review"),
    (["Randomized Controlled Trial", "Journal Article"], "rct"),
    (["Practice Guideline", "Review"], "position_stand"),
    (["Guideline"], "position_stand"),
    (["Consensus Development Conference"], "position_stand"),
    (["Journal Article", "Review"], "narrative_review"),
    (["Journal Article", "Research Support, Non-U.S. Gov't"], None),
    ([], None),
])
def test_pubmed_types_map_to_designs_best_evidence_first(types, design):
    assert pubmed_design(types) == design


def test_every_design_has_a_tier():
    names = [d for d, _ in curate.PUBMED_DESIGNS] + [d for _, d in curate.TITLE_DESIGNS + curate.DESIGN_TIERS]
    assert set(names) | {curate.DEFAULT_DESIGN} == set(curate.DESIGN_WEIGHTS)
    assert (curate.DESIGN_WEIGHTS["position_stand"], curate.DESIGN_WEIGHTS["narrative_review"]) == (0.6, 0.4)
    score = lambda d: curate.Scorer().score_all([make("d", design=d)])[0].score
    assert score("meta_analysis") - score("narrative_review") == pytest.approx(0.35 * 0.6)


@pytest.mark.parametrize("title, text, design", [
    ("Small doses of training: A Narrative Review", "We cite each meta-analysis and randomized trial.", "narrative_review"),
    ("College Position Stand. Training for adults", "Randomized trials were pooled.", "position_stand"),
    ("International consensus on training", "A systematic review informed it.", "position_stand"),
    ("Sets and strength: A Review", "Every meta-analysis agrees.", "narrative_review"),
    ("Sets and strength: A Scoping Review", "One meta-analysis exists.", "systematic_review"),
    ("Sets and strength: A Systematic Review and Meta-Analysis", "", "meta_analysis"),
    ("Sets and strength: A Systematic Review", "No meta-analysis was possible.", "systematic_review"),
    ("Effects of sets on strength", "In this meta-analysis we pooled trials.", "meta_analysis"),
    ("Effects of sets on strength", "Men were randomized to groups.", "rct"),
    ("Effects of sets on strength", "", "other"),
])
def test_keyword_fallback_lets_the_title_decide_first(title, text, design):
    assert keyword_design(title, f"{title} {text}") == design


@pytest.mark.parametrize("title, design", [
    ("Training: A Narrative Review", "narrative_review"), ("ABC Position Stand: Training", "position_stand"),
    ("A Consensus Statement on Training", "position_stand"), ("Is there a consensus on training?", None),
    ("Training: A Systematic Review and Narrative Review", None), ("Training: A Review", None),
])
def test_declared_design_is_strict(title, design):
    assert declared_design(title) == design


def test_declared_title_beats_pubmed_which_beats_keywords():
    papers = [make("d1", "Sets: A Narrative Review", "1"), make("d2", "Position Stand. Sets", "2"),
              make("d3", "Sets and strength", "3", design="meta_analysis"), make("d4", "Sets", "4", design="cohort")]
    set_designs(papers, client(types={"1": (["Meta-Analysis", "Review"], "d1"), "2": (["Review"], None),
                                      "3": (["Randomized Controlled Trial"], "d3"), "4": (["Journal Article"], "d4")}), [])
    assert [(p.design, p.design_source) for p in papers] == [
        ("narrative_review", "keyword"), ("position_stand", "keyword"), ("rct", "pubmed"), ("cohort", "keyword")]
    assert papers[0].pubtypes == ["Meta-Analysis", "Review"]


def test_requests_are_batched_posts_with_tool_and_email():
    c = client(types={"1": (["Review"], "d1"), "2": (["Meta-Analysis"], "d2")})
    set_designs([make("d1", pmid="1"), make("d2", pmid="2")], c, [])
    (url, data), = c.session.calls
    assert url.endswith("esummary.fcgi") and data["id"] == "1,2"
    assert data["tool"] == "ScienceFit" and data["email"] == "me@example.com" and data["db"] == "pubmed"


def test_missing_pmid_is_looked_up_by_doi_for_manifest_papers_only():
    c = client(types={"9": (["Meta-Analysis"], "10.1/x")}, dois={'"10.1/x"[doi]': ["9"], '"10.1/two"[doi]': ["7", "8"]})
    known, ambiguous, fresh = (make(f"https://doi.org/10.1/{d}") for d in ("x", "two", "new"))
    set_designs([known, ambiguous, fresh], c, [{"doi": known.doi}, {"doi": ambiguous.doi}])
    assert (known.pmid, known.design) == ("9", "meta_analysis")
    assert ambiguous.pmid is None and fresh.pmid is None
    assert [d["term"] for u, d in c.session.calls if u.endswith("esearch.fcgi")] == ['"10.1/x"[doi]', '"10.1/two"[doi]']


def test_pmid_of_another_paper_or_unknown_pmid_is_dropped():
    wrong, unknown = make("https://doi.org/10.1/A", pmid="1", design="rct"), make("d2", pmid="2", design="rct")
    right = make("https://doi.org/10.1/B", pmid="3", design="rct")
    set_designs([wrong, unknown, right], client(types={"1": (["Review"], "10.1/other"), "3": (["Review"], "10.1/b")}), [])
    assert [(p.pmid, p.design) for p in (wrong, unknown, right)] == [(None, "rct"), (None, "rct"), ("3", "narrative_review")]


def test_cached_pubmed_types_are_reused_without_a_request():
    c, p = client(), make("d1", design="other")
    set_designs([p], c, [{"doi": "d1", "design_source": "pubmed", "pmid": "5", "pubtypes": ["Systematic Review"]}])
    assert (p.design, p.design_source, p.pmid, c.session.calls) == ("systematic_review", "pubmed", "5", [])


def test_retries_then_fails_without_exposing_the_email(monkeypatch):
    c = client(types={"1": (["Review"], None)}, status=[429, 500, 200])
    assert c.summaries(["1"])["1"]["pubtypes"] == ["Review"] and len(c.session.calls) == 3
    with pytest.raises(RuntimeError, match="PubMed HTTP 503") as e:
        client(status=[503] * 4).summaries(["1"])
    assert "example.com" not in str(e.value)
    broken = client()
    monkeypatch.setattr(broken.session, "post", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("email=me@example.com")))
    with pytest.raises(RuntimeError) as e:
        broken.summaries(["1"])
    assert "example.com" not in str(e.value) and e.value.__suppress_context__


def test_requests_are_spaced_at_under_three_per_second(monkeypatch):
    assert 1 / PubMedClient(email="", session=FakeNCBI()).delay < 3
    slept = []
    monkeypatch.setattr(curate.time, "sleep", slept.append)
    c = PubMedClient(email="", session=FakeNCBI(types={"1": (["Review"], None)}), delay=0.4)
    c.summaries(["1"]), c.summaries(["1"])
    assert slept[0] == 0.0 and 0.3 < slept[1] <= 0.4


def test_parser_reads_the_pmid_from_openalex():
    parse = curate.PaperParser().parse
    assert parse(raw_work("d", ids={"pmid": "https://pubmed.ncbi.nlm.nih.gov/123"})).pmid == "123"
    assert parse(raw_work("d")).pmid is None and "ids" in curate.FIELDS.split(",")


def test_curate_saves_pubmed_fields_and_keeps_the_manifest_when_pubmed_fails(tmp_path):
    path = tmp_path / "m.json"
    manifest.save([{"paper_id": "a_2020", "doi": "d1", "included": True}], path)
    before = path.read_text()
    works = FakeClient(known=[raw_work("d1", title="Resistance sets", ids={"pmid": "https://pubmed.ncbi.nlm.nih.gov/1"})])
    run = lambda pubmed: curate.curate([], download=False, search=False, client=works, pubmed=pubmed,
                                       papers_dir=tmp_path, manifest_path=path)
    with pytest.raises(RuntimeError):
        run(client(status=[500] * 4))
    assert path.read_text() == before
    run(client(types={"1": (["Meta-Analysis"], "d1")}))
    r = manifest.load(path)[0]
    assert (r["pmid"], r["pubtypes"], r["design"], r["design_source"]) == ("1", ["Meta-Analysis"], "meta_analysis", "pubmed")


@pytest.mark.parametrize("design, flags", [("narrative_review", [True, True, True]), ("meta_analysis", [False, False, True])])
def test_every_chunk_of_a_narrative_review_is_secondhand(tmp_path, monkeypatch, design, flags):
    monkeypatch.setattr(ingest, "embed", lambda texts: [[1.0, 0.5, 0.2] for _ in texts])
    store = ingest.VectorStore(path=tmp_path)
    store.add("a_2020", ["We found gains.", "12 3.4", "Doe et al. [6] reported gains."], {"title": "T", "design": design},
              ["results", "table", "discussion"])
    assert [h["secondhand"] for h in sorted(store.query("q"), key=lambda h: h["chunk_index"])] == flags
