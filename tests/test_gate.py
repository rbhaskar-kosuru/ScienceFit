"""Relevance gate, its audit log and the population tag. Titles and abstracts here are synthetic, written for the
tests (not copied from papers); each error class from evals/gate_labels.json has at least one case."""
import csv
import subprocess
import sys

import pytest
from src import config, curate, llm, manifest
from tests.test_pipeline import FakeClient, paper, raw_work

GATE = curate.RelevanceFilter()


def passes(text: str) -> bool:
    return GATE.check(text.lower()).passed


def abstract(*words: str) -> dict:
    """An OpenAlex inverted index for the given text."""
    index = {}
    for i, w in enumerate(" ".join(words).split()):
        index.setdefault(w, []).append(i)
    return index


# --- Error classes (synthetic text, one case per class found in evals/gate_labels.json) ---

@pytest.mark.parametrize("text", [
    "Insulin resistance and glucose tolerance after a low-carbohydrate diet",                  # "resistance" alone
    "Left ventricular hypertrophy in endurance-trained cyclists",                               # cardiac hypertrophy
    "Grading the strength of evidence for exercise recommendations in public health",          # strength of evidence
    "Antimicrobial resistance: a concise update",
    "Strength gain of cement composites with added portlandite",                                # lone "strength"
    "Myocardial hypertrophy after a year of higher training volume in divers",                  # lone "training volume"
])
def test_ambiguous_words_alone_no_longer_pass(text):
    assert not passes(text)


@pytest.mark.parametrize("text", [
    "Resistance training volume and hypertrophy: a randomized clinical trial in trained men",   # "clinical trial"
    "Protein supplementation during resistance training: clinical relevance for lifters",      # "clinical relevance"
    "Resistance exercise as therapy-free prevention: effects on blood pressure in healthy adults",  # "therapy"
    "Resistance training lowers cardiovascular disease risk factors in healthy middle-aged men",  # risk, not patients
    "Strength training and cancer prevention: a narrative review of myokines in healthy adults",  # bare "cancer"
])
def test_in_scope_wording_that_the_old_gate_rejected_now_passes(text):
    assert passes(text) and not curate.RelevanceFilter(*OLD_RULES).check(text.lower()).passed


def test_hypertrophy_counts_only_near_muscle_words():
    assert passes("Effects of rest intervals on hypertrophy of type II fibres")
    assert passes("Quadriceps muscle cross-sectional hypertrophy after eccentric loading")
    assert not passes("Hypertrophy of the prostate in older men")


@pytest.mark.parametrize("text", [
    "Resistance training in patients with chronic heart failure",
    "Creatine and resistance training in prostate cancer survivors receiving androgen deprivation",
    "Resistance training in hospitalized older adults after an acute illness",
    "Resistance exercise and muscle hypertrophy in rats",
    "Mechanical loading of C2C12 myotubes in vitro mimics resistance exercise",
    "Whey protein and resistance training in adults with type 2 diabetes",
    "Resistance training and creatine in older adults with mild cognitive impairment",
])
def test_clinical_animal_and_cell_studies_are_still_rejected(text):
    assert not passes(text)


@pytest.mark.parametrize("text", [
    "Calisthenics training improves push-up performance in schoolchildren",
    "Elastic band training in community-dwelling older women",
    "Kettlebell training and maximal strength in recreationally active men",
    "Suspension training versus free weights for upper-body strength",
    "Bodyweight exercises at home during lockdown and leg strength",
])
def test_every_kind_of_load_counts_as_resistance_training(text):
    assert passes(text)


@pytest.mark.parametrize("text, ok", [
    ("Higher protein intake during an energy deficit preserves lean mass in healthy young men", True),
    ("Leucine ingestion and muscle protein synthesis in healthy older adults", True),
    ("Pre-sleep casein and fat-free mass during a bulking phase", True),
    ("A low-fat diet versus a low-carbohydrate diet for weight loss in obesity: change in fat mass and body weight",
     False),
])
def test_nutrition_counts_with_a_muscle_outcome_but_not_weight_loss_alone(text, ok):
    assert passes(text) is ok


def test_guard_phrases_cannot_make_a_paper_pass_but_real_phrases_still_count():
    assert GATE.check("insulin resistance and cardiac hypertrophy").include == []
    assert GATE.check("insulin resistance fell after resistance training").include == ["resistance training"]


OLD_RULES = ({"resistance", "strength", "hypertrophy", "weightlifting", "1rm", "one-repetition", "periodization",
              "training volume"},
             {"patients", "patient", "disease", "hiv", "arthritis", "osteoarthritis", "rheumatoid", "cancer",
              "diabetes", "clinical", "metformin", "ibuprofen", "therapy", "mtorc1", "hydroxylation",
              "signaling pathway", "gene expression", "in vitro", "rodent", "mice", "murine", "knockout"}, (), ())


def test_eval_compares_against_the_same_old_gate():
    from evals.gate_eval import OLD_EXCLUDE, OLD_INCLUDE
    assert (OLD_INCLUDE, OLD_EXCLUDE) == OLD_RULES[:2]


# --- Audit log ---

def test_every_gate_decision_is_recorded_with_matched_terms(tmp_path):
    ranked = [curate.Paper(**{**vars(paper("A", d)), "text": t, "score": 0.9, "retracted": r})
              for d, t, r in [("in", "resistance training in young men", False),
                              ("rat", "resistance training in rats", False),
                              ("gone", "strength training study", True)]]
    curate.judge(ranked, target=0, cutoff=0.65, fetch=None, assign_id=lambda p: p.doi, papers_dir=tmp_path)
    got = {p.doi: (p.gate_pass, p.gate_include, p.gate_exclude, p.reason) for p in ranked}
    assert got == {
        "in": (True, ["resistance training"], [], "no PDF on disk (not downloaded)"),
        "rat": (False, ["resistance training"], ["rats"], "failed relevance gate"),
        "gone": (True, ["strength training"], [], "retracted"),  # judged even though retraction decides first
    }


def test_curate_writes_an_audit_of_every_candidate_including_rejections(tmp_path):
    works = FakeClient(fresh=[
        raw_work("d1", title="Resistance training volume in older adults", abstract_inverted_index=abstract("sets")),
        raw_work("d2", title="Insulin resistance in mice", abstract_inverted_index=abstract("glucose")),
    ])
    curate.curate(["q"], candidates=2, download=False, save=False, client=works, pubmed=NoPubMed(),
                  papers_dir=tmp_path, manifest_path=tmp_path / "m.json", audit_dir=tmp_path / "runs")
    (audit,) = (tmp_path / "runs").glob("gate_audit_*.csv")
    rows = {r["doi"]: r for r in csv.DictReader(audit.open())}
    assert set(rows) == {"d1", "d2"} and not (tmp_path / "m.json").exists()
    assert (rows["d1"]["gate_pass"], rows["d1"]["gate_include"], rows["d1"]["population"]) == \
        ("True", "resistance training", "older")
    assert (rows["d2"]["gate_pass"], rows["d2"]["gate_include"], rows["d2"]["gate_exclude"]) == ("False", "", "mice")
    assert list(rows["d1"]) == curate.AUDIT_COLUMNS


def test_default_audit_folder_is_gitignored():
    out = subprocess.check_output([sys.executable, "-c", "import src.config as c; print(c.GATE_AUDIT_DIR)"],
                                  cwd=config.ROOT, text=True)  # conftest points it at a temp folder in tests
    assert out.strip() == str(config.ROOT / "evals" / "runs")
    assert "evals/runs/" in (config.ROOT / ".gitignore").read_text().splitlines()


class NoPubMed:
    def find_pmid(self, doi):
        return None

    def summaries(self, pmids):
        return {}


# --- Population tag ---

@pytest.mark.parametrize("text, tag", [
    ("Plyometric training in adolescent soccer players", "youth"),
    ("Resistance training in children aged 8 to 12", "youth"),
    ("Resistance training in older adults aged 65 and over", "older"),
    ("Strength training in community-dwelling elderly women", "older"),  # "women" is part of an older group here
    ("Training to failure in resistance-trained men", "adult"),
    ("Protein intake in young adults and older adults", "mixed"),
    ("Effects of rest intervals on muscle hypertrophy", "unknown"),
    ("Creatine and resistance training in young men and older men", "mixed"),
    ("Strength gains in women and men", "adult"),
])
def test_population_tag_from_keywords(text, tag):
    assert curate.population(text) == tag


def test_population_is_parsed_and_saved_to_the_manifest(tmp_path):
    path = tmp_path / "m.json"
    manifest.save([{"paper_id": "a_2020", "doi": "d1", "included": True}], path)
    works = FakeClient(known=[raw_work("d1", title="Resistance training in older women")])
    curate.curate([], download=False, search=False, client=works, pubmed=NoPubMed(), papers_dir=tmp_path,
                  manifest_path=path)
    record = manifest.load(path)[0]
    assert (record["population"], record["gate_pass"], record["gate_include"]) == ("older", True, ["resistance training"])


# --- Token usage for eval budgets ---

def test_llm_client_keeps_the_token_usage_of_the_last_answer(monkeypatch):
    class Reply:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 42}}
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: Reply())
    client = llm.LLMClient(base_url="http://localhost:1/v1", model="m", api_key="")
    assert client.last_usage == {} and client.generate("s", "p") == "ok" and client.last_usage == {"total_tokens": 42}
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: type("R", (), {"status_code": 400, "text": "", "headers": {}})())
    with pytest.raises(llm.LLMError):
        client.generate("s", "p")
    assert client.last_usage == {}  # no stale count from the previous answer


# --- Labeller (evals/gate_label.py), with a fake LLM ---

def test_scope_covers_every_load_and_nutrition_with_muscle_outcomes():
    scope = config.CURATE_SCOPE.lower()
    for phrase in ("elastic", "bodyweight", "calisthenics", "kettlebell", "suspension", "lean mass",
                   "muscle protein synthesis", "energy deficit", "meal timing", "only fat or body weight"):
        assert phrase in scope
    from evals import gate_label
    assert config.CURATE_SCOPE in gate_label.SYSTEM


@pytest.mark.parametrize("item, again", [
    ({"label": "out", "population": "adult", "reason": "No resistance training; supplementation only"}, True),
    ({"label": "out", "population": "clinical", "reason": "Cancer patients"}, False),
    ({"label": "out", "population": "adult", "reason": "Animal study, not human"}, False),
    ({"label": "out", "population": "adult", "reason": "Study in rats"}, False),
    ({"label": "in", "population": "adult", "reason": "RT volume"}, False),  # the wider scope can't make it out
    ({"label": "out", "population": "adult", "reason": "No RT", "scope_version": 2}, False),  # already re-asked
])
def test_only_labels_the_wider_scope_could_flip_are_asked_again(item, again):
    from evals.gate_label import needs_relabel
    assert needs_relabel(item) is again


def test_labeller_parses_json_and_rejects_anything_else():
    from evals.gate_label import parse
    assert parse('x {"label": "IN", "population": "older", "reason": "ok"}') == \
        {"label": "in", "population": "older", "reason": "ok"}
    for bad in ("no json", '{"label": "maybe", "population": "adult"}', '{"label": "in", "population": "teen"}'):
        with pytest.raises(ValueError):
            parse(bad)


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.last_usage = list(replies), {}

    def generate(self, system, prompt):
        reply = self.replies.pop(0)
        self.last_usage = {}  # like LLMClient: a failed call reports no usage
        if isinstance(reply, Exception):
            raise reply
        self.last_usage = {"total_tokens": 100}
        return reply


def test_labeller_waits_out_per_minute_limits_but_stops_on_the_daily_one():
    from evals.gate_label import ask_patiently, over_budget, DAILY_BUDGET
    row, ok = {"title": "T", "text": "T abstract"}, '{"label": "in", "population": "adult", "reason": "r"}'
    labels, slept = {"tokens_used": 0}, []
    minute = llm.LLMError("LLM m: HTTP 429: Rate limit reached on tokens per minute (TPM)")
    assert ask_patiently(FakeLLM([minute, ok]), row, labels, sleep=slept.append)["label"] == "in"
    assert slept == [30] and labels["tokens_used"] == 100
    day = llm.LLMError("LLM m: HTTP 429: Rate limit reached on tokens per day (TPD)")
    with pytest.raises(llm.LLMError):
        ask_patiently(FakeLLM([day]), row, labels, sleep=slept.append)
    from datetime import date
    assert not over_budget(labels) and over_budget({"tokens_by_day": {str(date.today()): DAILY_BUDGET}})
