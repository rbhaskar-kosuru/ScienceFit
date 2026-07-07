# ScienceFit

A RAG chatbot that answers resistance-training questions grounded in peer-reviewed research, with inline citations to the source papers.

No bro-science. No YouTube guesses. Just retrieval over curated studies.

## Why

Most fitness chatbots coach behavior. None retrieve from actual research papers and cite them back. ScienceFit fills that gap: ask a question, get an answer traceable to the study it came from.

## How it works

```
question → embed → similarity search (Chroma)
        → top-k paper chunks → LLM (Ollama)
        → answer with [paper_id] citations
```

The LLM never sees all papers — only the top-k chunks the retriever picks. Swap the LLM without touching the retriever.

## Stack

| Layer      | Tool                                  |
|------------|---------------------------------------|
| LLM        | Ollama (`llama3.1`) — local, free     |
| Embeddings | Ollama (`nomic-embed-text`)           |
| Vector DB  | Chroma (persistent, local)            |
| Backend    | FastAPI                               |
| Frontend   | Streamlit                             |
| PDF parse  | pypdf                                 |

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- ~20 research PDFs on resistance training (curated, not scraped)

## Setup

```bash
git clone https://github.com/rbhaskar-kosuru/ScienceFit.git
cd ScienceFit

# Install Ollama (one-time)
# macOS:   brew install ollama
# Linux:   curl -fsSL https://ollama.com/install.sh | sh
# Windows: download from https://ollama.com/download

# Start Ollama 
ollama serve &
# Open a new terminal and Pull models
ollama pull llama3.1
ollama pull nomic-embed-text
ollama list # both should show in the list


# Python env
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. **Drop PDFs** into `data/papers/`. Filename becomes the citation key, e.g. `schoenfeld_2017.pdf` → `[schoenfeld_2017]`.

2. **Ingest** — chunk, embed, store in Chroma:
   ```bash
   python -m src.ingest
   ```

3. **Run API** (terminal 1):
   ```bash
   uvicorn src.api:app --reload
   ```

4. **Run UI** (terminal 2):
   ```bash
   streamlit run app.py
   ```

Open the Streamlit URL and ask something like *"What's the optimal weekly set volume for hypertrophy?"*

### API only

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "optimal reps for hypertrophy?", "top_k": 5}'
```

## Project structure

```
ScienceFit/
├── app.py                # Streamlit UI
├── requirements.txt
├── data/
│   ├── papers/           # your PDFs (gitignored)
│   └── chroma/           # vector store (gitignored)
└── src/
    ├── config.py         # paths, models, chunk sizes
    ├── ingest.py         # PaperLoader, Chunker, VectorStore
    ├── rag.py            # RAG: retrieve → prompt → generate
    └── api.py            # FastAPI endpoints
```

## Configuration

Tune in `src/config.py`:

| Setting         | Default              | Notes                              |
|-----------------|----------------------|------------------------------------|
| `LLM_MODEL`     | `llama3.1`           | Any Ollama model                   |
| `EMBED_MODEL`   | `nomic-embed-text`   | Must match at ingest and query time|
| `CHUNK_SIZE`    | 800                  | Characters per chunk               |
| `CHUNK_OVERLAP` | 150                  | Preserves context across splits    |
| `TOP_K`         | 5                    | Chunks passed to LLM               |

## Swapping the LLM

To use Claude, GPT-4, or any hosted model, change `_generate` in `src/rag.py`. The retriever stays the same.


## Notes

- Answers are only as good as the papers you feed it. Curate carefully.
- Papers I personally curated are used for research/educational purposes, not redistributed.
- The LLM can still misinterpret excerpts. Citations let users verify.
- Not medical or professional advice.


## License

MIT
