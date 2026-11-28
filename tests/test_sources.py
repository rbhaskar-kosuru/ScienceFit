"""Evidence labels on sources: rag.sources(), the /ask schema and the UI. Store, reranker, LLM and API are faked."""
import importlib
import sys

import pytest
import requests
from src import config, rag
from src.rag import RAG, evidence
from tests.test_pipeline import FakeReranker, FakeStore, hit


def ask(*hits, answer="Yes [a_2020] [b_2021]."):
    return RAG(FakeStore(list(hits)), FakeReranker([3.0] * len(hits)), lambda s, p: answer).ask("How many sets?")


def by_id(res):
    return {s["paper_id"]: s for s in res["sources"]}


# --- Rules ---

EVIDENCE = {"primary", "narrative_review", "position_stand"}


@pytest.mark.parametrize("design", ["narrative_review", "position_stand"])
def test_review_designs_are_labelled_by_design_whatever_their_chunks(design):
    for flag in (True, False):
        src = by_id(ask(hit("a_2020", 0, design=design, secondhand=flag)))["a_2020"]
        assert (src["evidence"], src["secondhand_passage"]) == (design, False)  # reviews don't get the passage flag


def test_primary_paper_with_a_secondhand_passage_is_flagged():
    res = ask(hit("a_2020", 0, design="rct", secondhand=True), hit("a_2020", 1, design="rct", secondhand=False))
    src = by_id(res)["a_2020"]
    assert (src["evidence"], src["secondhand_passage"]) == ("primary", True)


def test_primary_paper_without_a_secondhand_passage_is_not_flagged():
    res = ask(hit("a_2020", 0, design="rct", secondhand=False), hit("a_2020", 1, design="rct", secondhand=False))
    src = by_id(res)["a_2020"]
    assert (src["evidence"], src["secondhand_passage"]) == ("primary", False)


def test_all_secondhand_chunks_no_longer_change_the_paper_label():
    res = ask(hit("a_2020", 0, design="rct", secondhand=True), hit("a_2020", 1, design="rct", secondhand=True))
    src = by_id(res)["a_2020"]
    assert (src["evidence"], src["secondhand_passage"]) == ("primary", True)
    assert "second_hand" not in EVIDENCE


@pytest.mark.parametrize("design", ["meta_analysis", "systematic_review"])
def test_meta_analyses_and_systematic_reviews_are_primary(design):
    src = by_id(ask(hit("a_2020", 0, design=design, secondhand=True)))["a_2020"]
    assert (src["evidence"], src["secondhand_passage"]) == ("primary", True)
    assert evidence(design) == "primary"


def test_missing_design_or_flag_is_primary_without_passage_flag():
    assert evidence(None) == "primary"
    src = by_id(ask(hit("a_2020", 0)))["a_2020"]  # no design, no secondhand flag (pre-v4 index)
    assert (src["evidence"], src["secondhand_passage"]) == ("primary", False)
    assert src["chunks"] == [{"index": 0, "section": "unknown", "secondhand": False}]


def test_labels_are_per_paper_and_chunks_carry_their_flag():
    res = ask(hit("a_2020", 0, design="narrative_review", secondhand=True),
              hit("b_2021", 4, design="rct", section="discussion", secondhand=True),
              hit("b_2021", 2, design="rct", section="results", secondhand=False))
    a, b = by_id(res)["a_2020"], by_id(res)["b_2021"]
    assert (a["evidence"], b["evidence"]) == ("narrative_review", "primary")
    assert (a["secondhand_passage"], b["secondhand_passage"]) == (False, True)
    assert b["chunks"] == [{"index": 4, "section": "discussion", "secondhand": True},
                           {"index": 2, "section": "results", "secondhand": False}]


def test_refused_answers_are_unchanged():
    refused = RAG(FakeStore([hit("a_2020", secondhand=True)]), FakeReranker([-10.0]), None).ask("How many sets?")
    assert refused == {"answer": rag.NO_EVIDENCE, "refused": True, "sources": [], "unverified_citations": [],
                       "no_citations": False}


# --- API schema ---

@pytest.fixture
def api(monkeypatch):
    """src.api with a RAG built on fakes: no Chroma, reranker or LLM."""
    hits = [hit("a_2020", 0, design="narrative_review", secondhand=True, title="A"),
            hit("b_2021", 1, design="meta_analysis", secondhand=True, section="discussion"),
            hit("c_2022", 2, design="position_stand", secondhand=False, title="C")]
    monkeypatch.setattr(rag, "RAG", lambda: RAG(FakeStore(hits), FakeReranker([3.0, 2.0, 1.0]), lambda s, p: "Yes [a_2020]."))
    sys.modules.pop("src.api", None)
    from fastapi.testclient import TestClient
    module = importlib.import_module("src.api")
    yield TestClient(module.app)
    sys.modules.pop("src.api", None)


def test_ask_returns_evidence_per_source_and_secondhand_per_chunk(api):
    body = api.post("/ask", json={"question": "How many sets?"}).json()
    assert set(body) == {"answer", "refused", "sources", "unverified_citations", "no_citations"}
    for src in body["sources"]:
        assert src["evidence"] in EVIDENCE and isinstance(src["secondhand_passage"], bool)
        assert all(set(c) == {"index", "section", "secondhand"} and isinstance(c["secondhand"], bool)
                   for c in src["chunks"])
    assert {s["paper_id"]: (s["evidence"], s["secondhand_passage"]) for s in body["sources"]} == {
        "a_2020": ("narrative_review", False), "b_2021": ("primary", True), "c_2022": ("position_stand", False)}


# --- UI ---

class FakeAPI:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        def source(pid, cited, **fields):
            return {"paper_id": pid, "title": pid.upper(), "cited": cited, "relevance": 0.9, "chunks": [], **fields}
        return {"answer": "Yes [nar_2020].", "refused": False, "unverified_citations": [], "no_citations": False,
                "sources": [source("nar_2020", True, evidence="narrative_review", secondhand_passage=False),
                            source("pos_2021", False, evidence="position_stand", secondhand_passage=False),
                            source("prs_2022", False, evidence="primary", secondhand_passage=True),
                            source("pri_2023", False, evidence="primary", secondhand_passage=False),
                            source("old_2019", False),  # older API: neither field
                            source("lab_2018", False, evidence="second_hand")]}  # value this API no longer sends


def test_ui_labels_sources_from_the_api_fields(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeAPI())
    app = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=20).run()
    app.chat_input[0].set_value("How many sets?").run()
    assert not app.exception
    lines = {line.split("**[")[1].split("]")[0]: line for m in app.markdown for line in m.value.split("\n- ")
             if "**[" in line}
    assert lines["nar_2020"].endswith("**✓ cited** · _narrative review_")
    assert lines["pos_2021"].endswith("_retrieved, not cited_ · _position stand_")
    assert lines["prs_2022"].endswith("_retrieved, not cited_ · _this passage retells other studies_")
    for pid in ("pri_2023", "old_2019", "lab_2018"):  # nothing extra
        assert lines[pid].endswith("_retrieved, not cited_")


def test_ui_has_no_backend_imports():
    source = (config.ROOT / "app.py").read_text()
    assert "src." not in source and "from src" not in source and "import src" not in source
