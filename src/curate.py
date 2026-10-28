"""Curation: OpenAlex search → relevance gate → score → rank → download open-access PDFs → manifest."""
import re
import time
import argparse
import unicodedata
from datetime import date
from dataclasses import dataclass, asdict, field

import requests

from . import config, manifest

OPENALEX = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "ScienceFit/0.2 (research RAG)"}

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
    openalex_id: str
    doi: str | None
    title: str
    first_author: str
    year: int | None
    journal: str
    citations: int
    citations_per_year: float
    design: str
    is_oa: bool
    pdf_url: str | None
    retracted: bool
    score: float = 0.0
    paper_id: str = ""
    pdf_file: str | None = None
    text: str = field(default="", repr=False)

    def key(self) -> str:
        return (self.doi or self.openalex_id or self.title).lower()

    def to_record(self) -> dict:
        record = asdict(self)
        record.pop("text")
        return record


class OpenAlexClient:
    def __init__(self, mailto: str = config.OPENALEX_MAILTO):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.mailto = mailto

    def search(self, query: str, n: int = 100, from_year: int = 2005) -> list[dict]:
        results, per_page = [], 50
        for page in range(1, (n + per_page - 1) // per_page + 1):
            params = {
                "search": query,
                "filter": f"from_publication_date:{from_year}-01-01,type:article",
                "per-page": per_page,
                "page": page,
            }
            if self.mailto:
                params["mailto"] = self.mailto  # OpenAlex "polite pool": faster, more reliable
            r = self.session.get(OPENALEX, params=params, timeout=30)
            r.raise_for_status()
            results.extend(r.json().get("results", []))
            time.sleep(0.2)
        return results[:n]


class PaperParser:
    def __init__(self, current_year: int | None = None):
        self.current_year = current_year or date.today().year

    def parse(self, raw: dict) -> Paper:
        year = raw.get("publication_year")
        citations = raw.get("cited_by_count", 0)
        title = raw.get("title") or ""
        blob = f"{title} {self._abstract(raw.get('abstract_inverted_index'))}"
        oa = raw.get("open_access") or {}
        best = raw.get("best_oa_location") or {}
        return Paper(
            openalex_id=raw.get("id", ""),
            doi=raw.get("doi"),
            title=title,
            first_author=self._first_author(raw),
            year=year,
            journal=self._journal(raw),
            citations=citations,
            citations_per_year=self._per_year(citations, year),
            design=self._design(blob)[0],
            is_oa=oa.get("is_oa", False),
            pdf_url=best.get("pdf_url") or oa.get("oa_url"),  # direct PDF link beats a landing page
            retracted=raw.get("is_retracted", False),
            text=blob.lower(),
        )

    def _per_year(self, citations: int, year: int | None) -> float:
        if not year:
            return 0.0
        return round(citations / max(self.current_year - year + 1, 1), 2)

    @staticmethod
    def _first_author(raw: dict) -> str:
        authors = raw.get("authorships") or []
        return ((authors[0].get("author") or {}).get("display_name") or "").strip() if authors else ""

    @staticmethod
    def _journal(raw: dict) -> str:
        source = (raw.get("primary_location") or {}).get("source") or {}
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


class IdAssigner:
    """Readable, stable citation keys: schoenfeld_2017, schoenfeld_2017b, ... Same paper keeps its ID across runs."""

    def __init__(self, existing: dict[str, str]):
        self.existing = existing  # paper key (DOI or OpenAlex ID) -> paper_id
        self.taken = set(existing.values())

    def assign(self, p: Paper) -> str:
        if p.key() in self.existing:
            return self.existing[p.key()]
        base = citation_key(p.first_author, p.year, p.title)
        pid, suffix = base, ord("b")
        while pid in self.taken:
            pid, suffix = f"{base}{chr(suffix)}", suffix + 1
        self.taken.add(pid)
        self.existing[p.key()] = pid
        return pid


def citation_key(author: str, year: int | None, title: str) -> str:
    words = _ascii(author or title).split()
    name = re.sub(r"[^a-z0-9]", "", words[-1 if author else 0].lower()) if words else ""
    name = name or "paper"
    return f"{name}_{year}" if year else name


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def dedupe(papers: list[Paper]) -> list[Paper]:
    seen, unique = set(), []
    for p in papers:
        if p.key() not in seen:
            seen.add(p.key())
            unique.append(p)
    return unique


class PDFDownloader:
    def __init__(self, out_dir=config.PAPERS_DIR):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def download(self, papers: list[Paper]) -> list[Paper]:
        return [p for p in papers if p.pdf_url and self._fetch(p)]

    def _fetch(self, p: Paper) -> bool:
        target = self.out_dir / f"{p.paper_id}.pdf"
        if not target.exists():
            try:
                r = self.session.get(p.pdf_url, timeout=60)
            except requests.RequestException:
                return False
            if not (r.ok and r.content[:5] == b"%PDF-"):  # landing pages and paywalls return HTML
                return False
            target.write_bytes(r.content)
        p.pdf_file = target.name
        return True


def curate(query: str, pull: int = 100, keep: int = 20, download: bool = True) -> list[Paper]:
    raw = OpenAlexClient().search(query, n=pull)
    parser = PaperParser()
    papers = dedupe([parser.parse(r) for r in raw])
    relevant = RelevanceFilter().apply(papers)
    print(f"Relevance gate: {len(relevant)}/{len(papers)} passed.")
    ranked = Scorer().score_all(relevant)[:keep]

    old = manifest.load()
    ids = IdAssigner({(r.get("doi") or r.get("openalex_id") or r.get("title", "")).lower(): r["paper_id"] for r in old})
    for p in ranked:
        p.paper_id = ids.assign(p)
    if download:
        got = PDFDownloader().download(ranked)
        print(f"PDFs available: {len(got)}/{len(ranked)} (paywalled papers stay in the manifest, no PDF).")

    records = manifest.merge(old, [p.to_record() for p in ranked])
    manifest.save(records)
    print(f"Manifest: {config.MANIFEST_PATH} ({len(records)} papers)")
    return ranked


def main() -> None:
    ap = argparse.ArgumentParser(description="Curate resistance-training papers from OpenAlex.")
    ap.add_argument("query", nargs="?", default="resistance training hypertrophy muscle strength")
    ap.add_argument("--pull", type=int, default=100, help="papers to fetch from OpenAlex")
    ap.add_argument("--keep", type=int, default=20, help="top-ranked papers to keep")
    ap.add_argument("--no-download", action="store_true", help="rank only, skip PDF downloads")
    args = ap.parse_args()

    ranked = curate(args.query, pull=args.pull, keep=args.keep, download=not args.no_download)
    for i, p in enumerate(ranked, 1):
        pdf = "PDF" if p.pdf_file else "   "
        print(f"{i:2d}. [{p.score:.3f}] {pdf} {p.paper_id:22s} {p.design:16s} {p.title[:60]}")


if __name__ == "__main__":
    main()
