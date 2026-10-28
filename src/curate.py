"""Curation: OpenAlex search → open-access + relevance gate → score → cutoff → download PDFs in rank order → manifest."""
import re
import math
import time
import argparse
import unicodedata
from datetime import date
from urllib.parse import urlparse
from dataclasses import dataclass, asdict, field

import requests

from . import config, manifest

OPENALEX = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "ScienceFit/0.2 (research RAG)"}
FIELDS = ("id,doi,title,publication_year,cited_by_count,authorships,primary_location,"
          "open_access,best_oa_location,locations,is_retracted,abstract_inverted_index")

# Design tiers scanned from title/abstract, best evidence first.
DESIGN_TIERS = [
    (("meta-analysis", "meta analysis"), "meta_analysis", 1.0),
    (("systematic review",), "systematic_review", 0.9),
    (("randomized", "randomised", "rct"), "rct", 0.8),
    (("cohort", "longitudinal", "prospective"), "cohort", 0.5),
    (("cross-sectional", "observational"), "observational", 0.4),
]
DEFAULT_DESIGN = ("other", 0.3)

# Q1/Q2 sports-science venues (proxy for Scimago quartile without bundling the CSV). Stored normalized.
TRUSTED_JOURNALS = {
    "sports medicine",
    "medicine and science in sports and exercise",
    "journal of strength and conditioning research",
    "european journal of applied physiology",
    "journal of sports sciences",
    "scandinavian journal of medicine and science in sports",
    "frontiers in physiology",
    "peerj",
    "sports",
}

WEIGHTS = {"citations": 0.4, "design": 0.35, "journal": 0.25}

# Relevance gate. Paper must mention >=1 resistance-specific term and 0 excluded terms.
# "training" and "muscle" alone are intentionally NOT here: endurance, cognitive and molecular papers all contain them.
DOMAIN_TERMS = {
    "resistance", "strength", "hypertrophy", "weightlifting",
    "1rm", "one-repetition", "periodization", "training volume",
}
# Clinical populations + molecular/mechanistic bench-science markers.
CLINICAL_TERMS = {
    "patients", "patient", "disease", "hiv", "arthritis", "osteoarthritis", "rheumatoid",
    "cancer", "diabetes", "clinical", "metformin", "ibuprofen", "therapy",
    "mtorc1", "hydroxylation", "signaling pathway", "gene expression",
    "in vitro", "rodent", "mice", "murine", "knockout",
}


def term_pattern(terms) -> re.Pattern:
    """Whole-word match that skips negated forms: 'sets' misses 'subsets', 'randomized' misses 'non-randomized'."""
    alternatives = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<!non-)(?<!non )\b(?:{alternatives})\b")


DESIGN_PATTERNS = [(term_pattern(keywords), name, weight) for keywords, name, weight in DESIGN_TIERS]


def normalize_journal(name: str) -> str:
    """'Medicine & Science in Sports & Exercise' -> 'medicine and science in sports and exercise'."""
    name = name.lower().replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", name).split())


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
    pdf_urls: list[str] = field(default_factory=list)  # every OA PDF link, best first; fallbacks if one fails
    score: float = 0.0
    paper_id: str = ""
    pdf_file: str | None = None
    text: str = field(default="", repr=False)

    def key(self) -> str:
        return (self.doi or self.openalex_id or self.title).lower()

    def to_record(self) -> dict:
        record = asdict(self)
        record.pop("text")
        record.pop("pdf_urls")
        return record


class OpenAlexClient:
    def __init__(self, mailto: str = config.OPENALEX_MAILTO, api_key: str = config.OPENALEX_API_KEY,
                 session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)
        self.mailto = mailto
        self.api_key = api_key

    def search(self, query: str, n: int = 200, from_year: int = config.CURATE_FROM_YEAR) -> list[dict]:
        results, per_page = [], min(n, 200)
        for page in range(1, math.ceil(n / per_page) + 1):
            params = {
                "search": query,
                "filter": f"from_publication_date:{from_year}-01-01,type:article|review,is_oa:true",
                "select": FIELDS,
                "per-page": per_page,
                "page": page,
            }
            if self.mailto:
                params["mailto"] = self.mailto  # OpenAlex "polite pool"
            if self.api_key:
                params["api_key"] = self.api_key
            batch = self._get(params).get("results", [])
            results.extend(batch)
            if len(batch) < per_page:
                break  # ran out of matches
            time.sleep(0.2)
        return results[:n]

    def search_many(self, queries: list[str], total: int) -> list[dict]:
        """Split `total` evenly across queries. Overlap between queries is removed later by dedupe()."""
        per_query = math.ceil(total / len(queries))
        return [raw for q in queries for raw in self.search(q, n=per_query)]

    def _get(self, params: dict, retries: int = 4) -> dict:
        for attempt in range(retries):
            r = self.session.get(OPENALEX, params=params, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(float(r.headers.get("Retry-After") or 2 ** attempt))
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return {}


class PaperParser:
    def __init__(self, current_year: int | None = None):
        self.current_year = current_year or date.today().year

    def parse(self, raw: dict) -> Paper:
        year = raw.get("publication_year")
        citations = raw.get("cited_by_count", 0)
        title = raw.get("title") or ""
        blob = f"{title} {self._abstract(raw.get('abstract_inverted_index'))}"
        oa = raw.get("open_access") or {}
        urls = self._pdf_urls(raw)
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
            pdf_url=urls[0] if urls else None,
            retracted=raw.get("is_retracted", False),
            pdf_urls=urls,
            text=blob.lower(),
        )

    def _per_year(self, citations: int, year: int | None) -> float:
        if not year:
            return 0.0
        return round(citations / max(self.current_year - year + 1, 1), 2)

    @staticmethod
    def _pdf_urls(raw: dict) -> list[str]:
        """Direct PDF links beat landing pages: best OA location first, then other OA locations, then oa_url."""
        best = raw.get("best_oa_location") or {}
        others = [loc.get("pdf_url") for loc in raw.get("locations") or [] if loc.get("is_oa")]
        candidates = [best.get("pdf_url"), *others, (raw.get("open_access") or {}).get("oa_url")]
        return list(dict.fromkeys(u for u in candidates if u))

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
        for pattern, name, weight in DESIGN_PATTERNS:
            if pattern.search(low):
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
        self.domain = term_pattern(domain)
        self.clinical = term_pattern(clinical)

    def apply(self, papers: list[Paper]) -> list[Paper]:
        return [p for p in papers if self._relevant(p)]

    def _relevant(self, p: Paper) -> bool:
        return bool(self.domain.search(p.text)) and not self.clinical.search(p.text)


class Scorer:
    def __init__(self, weights: dict = WEIGHTS, citation_cap: float = config.CITATION_CAP,
                 min_score: float | None = config.CURATE_MIN_SCORE):
        self.weights = weights
        self.citation_cap = citation_cap
        self.min_score = min_score

    def score_all(self, papers: list[Paper]) -> list[Paper]:
        valid = [p for p in papers if not p.retracted]
        for p in valid:
            p.score = round(self._score(p), 4)
        if self.min_score is not None:
            valid = [p for p in valid if p.score >= self.min_score]
        return sorted(valid, key=lambda p: p.score, reverse=True)

    def _score(self, p: Paper) -> float:
        # Fixed log scale, not relative to the pool, so a score means the same thing on every run.
        citation = min(math.log1p(p.citations_per_year) / math.log1p(self.citation_cap), 1.0)
        design = self._design_weight(p.design)
        journal = 1.0 if normalize_journal(p.journal) in TRUSTED_JOURNALS else 0.5
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
    def __init__(self, out_dir=config.PAPERS_DIR, session: requests.Session | None = None, host_delay: float = 1.0):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)
        self.host_delay = host_delay  # seconds between hits on the same host; publishers block bursts
        self.last_hit: dict[str, float] = {}

    def download_until(self, papers: list[Paper], target: int, assign_id,
                       max_attempts: int = config.CURATE_MAX_DOWNLOAD_ATTEMPTS) -> list[Paper]:
        """Walk papers in rank order until `target` valid PDFs. IDs are assigned only to papers we try."""
        got, attempts = [], 0
        for p in papers:
            if len(got) >= target or attempts >= max_attempts:
                break
            attempts += 1
            p.paper_id = assign_id(p)
            reason = self._fetch(p)
            if reason:
                print(f"  skip {p.paper_id}: {reason}")
            else:
                got.append(p)
        if len(got) < target:
            print(f"Warning: only {len(got)}/{target} PDFs after {attempts} attempts.")
        return got

    def _fetch(self, p: Paper) -> str | None:
        """Try each PDF link in turn. Returns None on success, else why the last link failed."""
        target = self.out_dir / f"{p.paper_id}.pdf"
        if target.exists():
            p.pdf_file = target.name
            return None
        reason = "no PDF link"
        for url in p.pdf_urls:
            self._wait(url)
            try:
                r = self.session.get(url, timeout=30)
            except requests.RequestException as e:
                reason = type(e).__name__
                continue
            if not r.ok:
                reason = f"HTTP {r.status_code}"
            elif r.content[:5] != b"%PDF-":  # landing pages and paywalls return HTML
                reason = "not a PDF"
            else:
                target.write_bytes(r.content)
                p.pdf_file = target.name
                return None
        return reason

    def _wait(self, url: str) -> None:
        host = urlparse(url).netloc
        gap = time.monotonic() - self.last_hit.get(host, -math.inf)
        if gap < self.host_delay:
            time.sleep(self.host_delay - gap)
        self.last_hit[host] = time.monotonic()


def curate(queries: list[str], candidates: int = config.CURATE_CANDIDATES, target: int = config.CURATE_TARGET_PDFS,
           download: bool = True) -> list[Paper]:
    raw = OpenAlexClient().search_many(queries, total=candidates)
    parser = PaperParser()
    papers = dedupe([parser.parse(r) for r in raw])
    print(f"Fetched: {len(raw)} results, {len(papers)} unique.")
    papers = [p for p in papers if p.is_oa and p.pdf_urls]
    print(f"Open access with a PDF link: {len(papers)}.")
    relevant = RelevanceFilter().apply(papers)
    print(f"Relevance gate: {len(relevant)}/{len(papers)} passed.")
    ranked = Scorer().score_all(relevant)
    cutoff = config.CURATE_MIN_SCORE
    print(f"Ranked: {len(ranked)}" + (f" with score >= {cutoff}." if cutoff is not None else " (no min score set)."))
    if not download:
        return ranked  # rank only: no IDs, no files, manifest untouched

    old = manifest.load()
    ids = IdAssigner({(r.get("doi") or r.get("openalex_id") or r.get("title", "")).lower(): r["paper_id"] for r in old})
    got = PDFDownloader().download_until(ranked, target, ids.assign)
    print(f"PDFs: {len(got)}/{target}.")

    # Only papers with a PDF are kept, so paywalled entries from older runs drop out too.
    records = [r for r in manifest.merge(old, [p.to_record() for p in got]) if r.get("pdf_file")]
    manifest.save(records)
    print(f"Manifest: {config.MANIFEST_PATH} ({len(records)} papers)")
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description="Curate resistance-training papers from OpenAlex.")
    ap.add_argument("query", nargs="?", help="single search query (default: config.CURATE_QUERIES)")
    ap.add_argument("--candidates", type=int, default=config.CURATE_CANDIDATES, help="papers to fetch from OpenAlex")
    ap.add_argument("--target", type=int, default=config.CURATE_TARGET_PDFS, help="valid PDFs to download")
    ap.add_argument("--no-download", action="store_true", help="rank only: no downloads, manifest untouched")
    args = ap.parse_args()

    queries = [args.query] if args.query else config.CURATE_QUERIES
    papers = curate(queries, candidates=args.candidates, target=args.target, download=not args.no_download)
    for i, p in enumerate(papers[:args.target], 1):
        pdf = "PDF" if p.pdf_file else "   "
        print(f"{i:2d}. [{p.score:.3f}] {pdf} {p.paper_id:22s} {p.design:16s} {p.title[:60]}")


if __name__ == "__main__":
    main()
