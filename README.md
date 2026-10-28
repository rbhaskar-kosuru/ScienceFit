# ScienceFit

A RAG chatbot that answers resistance-training questions grounded in peer-reviewed research, with citations you can verify.

No bro-science. No YouTube guesses. Just retrieval over curated studies.

## Why

Most fitness chatbots coach behavior. None retrieve from actual research papers and cite them back. ScienceFit fills that gap: ask a question, get an answer traceable to the study it came from — or an honest "no evidence" when the library doesn't cover it.

## How it works

```
Curate   OpenAlex search → relevance gate → score → rank → open-access PDFs + manifest
Ingest   PDFs + manifest → chunks → embeddings → Chroma (metadata on every chunk)
Ask      question → embed → top-20 candidates (Chroma)
                 → cross-encoder rerank → drop chunks below relevance threshold
                 → none left? refuse, no LLM call
                 → top-5 excerpts → LLM (Ollama) → answer
                 → citation validator → sources + flagged invented citations
```

- **Curation metadata survives ingest.** Title, year, journal, DOI, study design and quality score are stored on every chunk, so citations link to real papers.
- **Two-stage retrieval.** Vector search is fast but coarse; the cross-encoder reads question and chunk together and reorders them. Curation score breaks near-ties, so a meta-analysis edges out a weaker study.
- **Honest refusal.** If no chunk clears `MIN_RERANK_SCORE`, the API says so instead of guessing.
- **Citation validation.** Every `[paper_id]` in the answer is checked against the retrieved papers; anything invented is flagged.

## Stack

| Layer      | Tool                                                  |
|------------|-------------------------------------------------------|
| LLM        | Ollama (`llama3.2:3b`) — local, free                  |
| Embeddings | Ollama (`nomic-embed-text`)                           |
| Reranker   | `cross-encoder/ms-marco-MiniLM-L-6-v2` (Transformers) |
| Vector DB  | Chroma (persistent, local, cosine)                    |
| Data       | OpenAlex API                                          |
| Backend    | FastAPI                                               |
| Frontend   | Streamlit                                             |
| PDF parse  | pypdf                                                 |

Runs on an 8 GB Apple Silicon Mac. The reranker is ~90 MB and runs on CPU.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

## Setup

```bash
git clone https://github.com/rbhaskar-kosuru/ScienceFit.git
cd ScienceFit

# Install Ollama (one-time)
# macOS:   brew install ollama
# Linux:   curl -fsSL https://ollama.com/install.sh | sh
# Windows: download from https://ollama.com/download

# Start Ollama (macOS: or `brew services start ollama` to run it in the background)
ollama serve &

# In a new terminal, pull models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama list   # both should show

# Python env
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: your email for OpenAlex's faster "polite pool"
cp .env.example .env           # then edit OPENALEX_MAILTO
```

## Usage

1. **Curate papers** — fetch, score and rank from OpenAlex, download open-access PDFs:
   ```bash
   python -m src.curate                                  # default query, pull 100, keep 20
   python -m src.curate "training to failure" --keep 10  # custom query
   ```
   PDFs are saved as `data/papers/<paper_id>.pdf` with readable IDs like `schoenfeld_2017`. All ranked papers go into `data/curation_manifest.json`, including paywalled ones with no PDF. Re-running merges into the manifest and keeps existing IDs.

   You can also drop PDFs into `data/papers/` by hand. They're ingested with the filename as ID and no curation metadata.

2. **Ingest** — chunk, embed, store in Chroma:
   ```bash
   python -m src.ingest            # add or update papers
   python -m src.ingest --reset    # rebuild from scratch (after removing PDFs or changing models)
   ```

3. **Run API** (terminal 1):
   ```bash
   uvicorn src.api:app --reload
   ```
   The first start downloads the reranker (~90 MB) from Hugging Face.

4. **Run UI** (terminal 2):
   ```bash
   streamlit run app.py
   ```

Open the Streamlit URL and ask something like *"What's the optimal weekly set volume for hypertrophy?"*

### API

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "optimal reps for hypertrophy?", "top_k": 5}'
```

Response:

```json
{
  "answer": "… [schoenfeld_2017] …",
  "refused": false,
  "sources": [
    {"paper_id": "schoenfeld_2017", "title": "…", "year": 2017, "journal": "…",
     "doi": "https://doi.org/…", "design": "meta_analysis", "curated": true,
     "cited": true, "relevance": 0.94, "chunks": [12, 13]}
  ],
  "unverified_citations": []
}
```

`refused: true` means no excerpt cleared the relevance threshold, so no LLM call was made. `unverified_citations` lists tags the LLM cited that aren't in the retrieved papers.

### Tests

```bash
pytest
```

Ollama and the reranker are faked, so tests run offline.

## Project structure

```
ScienceFit/
├── app.py                # Streamlit UI
├── requirements.txt
├── .env.example          # OPENALEX_MAILTO
├── data/
│   ├── papers/           # PDFs (gitignored)
│   ├── chroma/           # vector store (gitignored)
│   └── curation_manifest.json  # paper metadata (gitignored)
├── src/
│   ├── config.py         # paths, models, retrieval knobs
│   ├── curate.py         # OpenAlex search → gate → score → rank → PDFs
│   ├── manifest.py       # metadata bridge between curate and ingest
│   ├── ingest.py         # PaperLoader, Chunker, VectorStore
│   ├── rerank.py         # cross-encoder reranker
│   ├── rag.py            # retrieve → rerank → refuse/generate → validate
│   └── api.py            # FastAPI endpoints
└── tests/
    └── test_pipeline.py
```

## Configuration

Tune in `src/config.py`:

| Setting            | Default                                | Notes                                           |
|--------------------|----------------------------------------|-------------------------------------------------|
| `LLM_MODEL`        | `llama3.2:3b`                          | Any Ollama model; no re-ingest needed           |
| `EMBED_MODEL`      | `nomic-embed-text`                     | Changing it requires `ingest --reset`           |
| `RERANK_MODEL`     | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Any Hugging Face cross-encoder                  |
| `CHUNK_SIZE`       | 800                                    | Characters per chunk                            |
| `CHUNK_OVERLAP`    | 150                                    | Preserves context across splits                 |
| `CANDIDATE_K`      | 20                                     | Vector-search candidates sent to the reranker   |
| `TOP_K`            | 5                                      | Max excerpts passed to the LLM                  |
| `MIN_RERANK_SCORE` | 0.1                                    | 0–1; below this a chunk is dropped              |
| `QUALITY_WEIGHT`   | 0.1                                    | How much curation score breaks relevance ties   |

## Swapping the LLM

To use Claude, GPT-4, Groq or any hosted model, pass a different `generate(system, prompt)` function to `RAG` (default: `ollama_generate` in `src/rag.py`). Retrieval, reranking and validation stay the same.

## Roadmap

- [x] MVP: RAG over curated papers with citations
- [x] Curation pipeline (OpenAlex, relevance gate, quality score)
- [x] Metadata carried through ingest
- [x] Cross-encoder reranking + honest refusal
- [x] Citation validation
- [ ] Evaluation harness (retrieval precision, answer faithfulness)
- [ ] Conversational memory (multi-turn)
- [ ] Dockerize
- [ ] Deploy: Railway (API) + Streamlit Cloud (UI)

## Notes

- Answers are only as good as the papers you feed it. Curate carefully.
- Only open-access PDFs download automatically; paywalled papers stay in the manifest for manual retrieval.
- Papers are used locally for retrieval and citation only, not redistributed.
- The LLM can still misread excerpts. Citations let users verify.
- Not medical or professional advice.

## License

MIT
