import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
CHROMA_DIR = DATA_DIR / "chroma"
MANIFEST_PATH = DATA_DIR / "curation_manifest.json"

# Models
LLM_MODEL = "llama3.2:3b"  # llama3.1 (8B) froze an 8GB M3
EMBED_MODEL = "nomic-embed-text"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# nomic-embed-text expects task prefixes. Changing them requires a re-ingest.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# Ingest
COLLECTION = "sciencefit_v2"  # bump when the chunk schema or embedding scheme changes
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBED_BATCH = 32

# Retrieval
CANDIDATE_K = 20        # vector-search candidates sent to the reranker
TOP_K = 5               # max reranked chunks passed to the LLM
MIN_RERANK_SCORE = 0.1  # 0-1; chunks below this are dropped, and if none pass the API refuses
QUALITY_WEIGHT = 0.1    # tie-break boost from the curation score (design, journal, citations)

# Curation
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "")
