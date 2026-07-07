from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "data" / "papers"
CHROMA_DIR = ROOT / "data" / "chroma"

COLLECTION = "sciencefit"
LLM_MODEL = "llama3.1"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
