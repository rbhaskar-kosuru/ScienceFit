"""Curation pipeline: search OpenAlex, score papers, rank, download top-N open-access PDFs."""
import re
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import requests

from . import config

OPENALEX = "https://api.openalex.org/works"

# Design tiers scanned from title/abstract, best evidence first.
DESIGN_TIERS = [
    (("meta-analysis", "meta analysis"), "meta_analysis", 1.0),
    (("systematic review",), "systematic_review", 0.9),
    (("randomized", "randomised", "rct"), "rct", 0.8),
    (("cohort", "longitudinal", "prospective"), "cohort", 0.5),
    (("cross-sectional", "observational"), "observational", 0.4),
]
DEFAULT_DESIGN = ("other", 0.3)

# Q1/Q2 sports-science venues (proxy for Scimago quartile without bundling the CSV).
TRUSTED_JOURNALS = {
    "sports medicine",
    "medicine and science in sports and exercise",
    "medicine science in sports exercise",
    "journal of strength and conditioning research",
    "european journal of applied physiology",
    "journal of sports sciences",
    "scandinavian journal of medicine science in sports",
    "frontiers in physiology",
    "peerj",
    "sports",
}

WEIGHTS = {"citations": 0.4, "design": 0.35, "journal": 0.25}

# Relevance gate. Paper must mention >=1 training-context term and 0 excluded terms.
# "muscle" alone is intentionally NOT here: molecular-biology papers all contain it.
DOMAIN_TERMS = {
    "training", "hypertrophy", "resistance training", "strength training",
    "sets", "reps", "repetition", "training volume", "resistance exercise",
    "weightlifting", "periodization", "1rm", "one-repetition",
}
# Clinical populations + molecular/mechanistic bench-science markers.
CLINICAL_TERMS = {
    "patients", "patient", "disease", "hiv", "arthritis", "rheumatoid",
    "cancer", "diabetes", "clinical", "metformin", "ibuprofen", "therapy",
    "mtorc1", "hydroxylation", "signaling pathway", "gene expression",
    "in vitro", "rodent", "mice", "murine", "knockout",
}


@dataclass
class Paper:
    id: str
    doi: str | None
    title: str
    year: int | None
    journal: str
    citations: int
    citations_per_year: float
    design: str
    is_oa: bool
    pdf_url: str | None
    retracted: bool
    text: str = ""
    score: float = 0.0


class OpenAlexClient:
    def __init__(self, mailto: str = "you@example.com"):
        self.session = requests.Session()
        self.mailto = mailto

    def search(self, query: str, n: int = 100, from_year: int = 2005) -> list[dict]:
        results, per_page = [], 50
        pages = (n + per_page - 1) // per_page
        for page in range(1, pages + 1):
            params = {
                "search": query,
                "filter": f"from_publication_date:{from_year}-01-01,type:article",
                "per-page": per_page,
                "page": page,
                "mailto": self.mailto,
            }
            r = self.session.get(OPENALEX, params=params, timeout=30)
            r.raise_for_status()
            results.extend(r.json().get("results", []))
            time.sleep(0.2)
        return results[:n]


class PaperParser:
    def __init__(self, current_year: int = 2026):
        self.current_year = current_year

    def parse(self, raw: dict) -> Paper:
        year = raw.get("publication_year")
        citations = raw.get("cited_by_count", 0)
        journal = self._journal(raw)
        title = raw.get("title") or ""
        abstract = self._abstract(raw.get("abstract_inverted_index"))
        blob = f"{title} {abstract}"
        design, _ = self._design(blob)
        oa = raw.get("open_access", {}) or {}
        return Paper(
            id=raw.get("id", ""),
            doi=raw.get("doi"),
            title=title,
            year=year,
            journal=journal,
            citations=citations,
            citations_per_year=self._per_year(citations, year),
            design=design,
            is_oa=oa.get("is_oa", False),
            pdf_url=oa.get("oa_url"),
            retracted=raw.get("is_retracted", False),
            text=blob.lower(),
        )

    def _per_year(self, citations: int, year: int | None) -> float:
        if not year:
            return 0.0
        return round(citations / max(self.current_year - year + 1, 1), 2)

    @staticmethod
    def _journal(raw: dict) -> str:
        loc = raw.get("primary_location") or {}
        source = loc.get("source") or {}
        return (source.get("display_name") or "").strip()

    @staticmethod
    def _design(text: str) -> tuple[str, float]:
        low = text.lower()
        for keywords, name, weight in DESIGN_TIERS:
            if any(k in low for k in keywords):
                return name, weight
        return DEFAULT_DESIGN

    @staticmethod
    def _abstract(inverted: dict | None) -> str:
        if not inverted:
            return ""
        positions = [(pos, word) for word, locs in inverted.items() for pos in locs]
        return " ".join(word for _, word in sorted(positions))


class RelevanceFilter:
    def __init__(self, domain: set = DOMAIN_TERMS, clinical: set = CLINICAL_TERMS):
        self.domain = domain
        self.clinical = clinical

    def apply(self, papers: list[Paper]) -> list[Paper]:
        return [p for p in papers if self._relevant(p)]

    def _relevant(self, p: Paper) -> bool:
        has_domain = any(term in p.text for term in self.domain)
        has_clinical = any(term in p.text for term in self.clinical)
        return has_domain and not has_clinical


class Scorer:
    def __init__(self, weights: dict = WEIGHTS):
        self.weights = weights

    def score_all(self, papers: list[Paper]) -> list[Paper]:
        valid = [p for p in papers if not p.retracted]
        max_cpy = max((p.citations_per_year for p in valid), default=1.0) or 1.0
        for p in valid:
            p.score = round(self._score(p, max_cpy), 4)
        return sorted(valid, key=lambda p: p.score, reverse=True)

    def _score(self, p: Paper, max_cpy: float) -> float:
        citation = p.citations_per_year / max_cpy
        design = self._design_weight(p.design)
        journal = 1.0 if p.journal.lower() in TRUSTED_JOURNALS else 0.5
        return (
            self.weights["citations"] * citation
            + self.weights["design"] * design
            + self.weights["journal"] * journal
        )

    @staticmethod
    def _design_weight(design: str) -> float:
        for _, name, weight in DESIGN_TIERS:
            if name == design:
                return weight
        return DEFAULT_DESIGN[1]


class PDFDownloader:
    def __init__(self, out_dir: Path = config.PAPERS_DIR):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def download(self, papers: list[Paper]) -> list[Paper]:
        got = []
        for p in papers:
            if p.pdf_url and self._fetch(p):
                got.append(p)
        return got

    def _fetch(self, p: Paper) -> bool:
        try:
            r = self.session.get(p.pdf_url, timeout=60)
            if r.ok and "pdf" in r.headers.get("content-type", "").lower():
                (self.out_dir / f"{self._slug(p)}.pdf").write_bytes(r.content)
                return True
        except requests.RequestException:
            pass
        return False

    @staticmethod
    def _slug(p: Paper) -> str:
        base = re.sub(r"[^a-z0-9]+", "_", p.title.lower()).strip("_")[:50]
        return base or p.id.rsplit("/", 1)[-1]


def curate(query: str, pull: int = 100, keep: int = 20, download: bool = True) -> list[Paper]:
    raw = OpenAlexClient().search(query, n=pull)
    parser = PaperParser()
    papers = [parser.parse(r) for r in raw]
    relevant = RelevanceFilter().apply(papers)
    print(f"Relevance gate: {len(relevant)}/{len(papers)} passed.")
    ranked = Scorer().score_all(relevant)[:keep]
    _save_manifest(ranked)
    if download:
        fetched = PDFDownloader().download(ranked)
        print(f"Downloaded {len(fetched)}/{len(ranked)} open-access PDFs.")
    return ranked


def _save_manifest(papers: list[Paper]) -> None:
    manifest = config.PAPERS_DIR.parent / "curation_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps([asdict(p) for p in papers], indent=2))
    print(f"Manifest written: {manifest} ({len(papers)} papers)")


if __name__ == "__main__":
    ranked = curate("resistance training hypertrophy muscle strength", pull=100, keep=20)
    for i, p in enumerate(ranked, 1):
        tag = "OA" if p.is_oa else "  "
        print(f"{i:2d}. [{p.score:.3f}] {tag} {p.design:16s} {p.citations_per_year:5.1f}/yr  {p.title[:70]}")
