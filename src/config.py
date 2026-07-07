from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "data" / "papers"
CHROMA_DIR = ROOT / "data" / "chroma"

COLLECTION = "sciencefit"
LLM_MODEL = "llama3.2:3b" # llama3.1 was too heavy to use on local machine
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
