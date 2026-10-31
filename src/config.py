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
LLM_TEMPERATURE = 0        # deterministic decoding so eval runs are repeatable
LLM_SEED = 42
EMBED_MODEL = "nomic-embed-text"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# nomic-embed-text expects task prefixes. Changing them requires a re-ingest.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# Ingest
COLLECTION = "sciencefit_v3"  # bump when the chunk schema or embedding scheme changes
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBED_BATCH = 32

# Retrieval
CANDIDATE_K = 20        # vector-search candidates sent to the reranker
TOP_K = 5               # max reranked chunks passed to the LLM
MIN_RERANK_SCORE = 0.1  # 0-1; chunks below this are dropped, and if none pass the API refuses
# Ranking uses the cross-encoder's raw logit (about -11 to +11); the 0-1 sigmoid is only for MIN_RERANK_SCORE.
QUALITY_WEIGHT = 0.5    # logit bonus x curation score (0-1): breaks near-ties, can't override a clearly better match
INTRO_PENALTY = 2.0     # logits subtracted from introduction chunks: they describe other studies
MAX_CHUNKS_PER_PAPER = 2  # cap per paper in the final TOP_K, so one paper can't crowd out the rest
# Questions under 3 words with none of these terms are refused before searching ("hi", "thanks").
FITNESS_TERMS = {
    "set", "rep", "repetition", "squat", "bench", "deadlift", "press", "protein", "creatine", "whey",
    "hypertrophy", "strength", "muscle", "training", "train", "workout", "lift", "lifting", "volume",
    "failure", "load", "intensity", "1rm", "cardio", "exercise", "supplement", "frequency", "rest",
}

# Curation
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")  # free key at openalex.org/settings/api; without it the IP shares $0.10/day
CURATE_QUERIES = [  # several angles so volume, frequency, load, etc. are all covered
    "resistance training hypertrophy",
    "training volume sets hypertrophy",
    "resistance training frequency",
    "load intensity strength gains",
    "proximity to failure resistance training",
    "rest interval resistance training",
    "protein supplementation resistance training",
    "creatine resistance training",
]
CURATE_CANDIDATES = 1000           # total pulled from OpenAlex, split evenly across queries
CURATE_TARGET_PDFS = 20            # new PDFs downloaded per run, best first; not a cap on the library
CURATE_MIN_SCORE = 0.65            # 0-1 quality cutoff before download; None = no cut
CURATE_FROM_YEAR = 2005
CURATE_MAX_DOWNLOAD_ATTEMPTS = 100 # safety cap on papers tried
CITATION_CAP = 50                  # citations/year that earns the full citation score (log-scaled)
