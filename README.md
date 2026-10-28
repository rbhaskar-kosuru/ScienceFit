# ScienceFit

A RAG chatbot that answers resistance-training questions grounded in peer-reviewed research, with citations you can verify.

No bro-science. No YouTube guesses. Just retrieval over curated studies.

## Why

Most fitness chatbots coach behavior. None retrieve from actual research papers and cite them back. ScienceFit fills that gap: ask a question, get an answer traceable to the study it came from — or an honest "no evidence" when the library doesn't cover it.

## How it works

```
Curate   new OpenAlex candidates + every manifest paper (re-fetched) → relevance gate → score
         → min-score cut → included if it has a valid PDF (up to 20 new downloads per run, best first)
         → manifest (included or excluded + reason)
Ingest   included papers → drop front matter and references → chunks → embeddings
         → rebuild Chroma from scratch (metadata on every chunk)
Ask      question → embed → top-20 candidates (Chroma)
                 → cross-encoder rerank → drop chunks below relevance threshold
                 → none left? refuse, no LLM call
                 → top-5 excerpts → LLM (Ollama) → answer
                 → citation validator → sources + flagged invented citations
```

- **Curation metadata survives ingest.** Title, year, journal, DOI, study design and quality score are stored on every chunk, so citations link to real papers.
- **Two-stage retrieval.** Vector search is fast but coarse; the cross-encoder reads question and chunk together and reorders them. Curation score breaks near-ties, so a meta-analysis edges out a weaker study.
- **Honest refusal.** If no chunk clears `MIN_RERANK_SCORE`, the API says so instead of guessing.
- **Citation validation.** Every `[paper_id]` in the answer is checked against the retrieved papers. Invented IDs and author-style citations (`[Cribb et al.]`, `[Smith 2019]`) are flagged; brackets that aren't citations (`[P < 0.05]`, `[n = 12]`, `[95% CI …]`) are ignored.

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

# Optional: OpenAlex email ("polite pool") and free API key
cp .env.example .env           # then edit OPENALEX_MAILTO / OPENALEX_API_KEY
```

## Usage

1. **Curate papers** — fetch, score and rank from OpenAlex, download open-access PDFs:
   ```bash
   python -m src.curate                                    # CURATE_QUERIES, ~1,000 candidates, 20 PDFs
   python -m src.curate --no-download                      # dry run: judge everything, no downloads, manifest untouched
   python -m src.curate --rejudge                          # no search or downloads: re-score manifest papers, update flags
   python -m src.curate "training to failure" --target 10  # single custom query
   ```
   Candidates are pulled 200 per page across `CURATE_QUERIES`, deduplicated by DOI and limited to open access. Articles and reviews (where OpenAlex files most meta-analyses) are fetched. A paper passes the relevance gate if it mentions a resistance-specific term (resistance, strength, hypertrophy, 1RM, …; `training` alone isn't enough) and no clinical or bench-science term. Matching is whole-word, so `strength` doesn't match `strengthen` and `randomized` doesn't match `non-randomized`. Score = 0.40 × citations (log scale, full marks at `CITATION_CAP`/year) + 0.35 × study design + 0.25 × trusted journal. Papers below `CURATE_MIN_SCORE` are cut.

   Every run judges **all** papers together: the new candidates plus every paper already in the manifest, re-fetched from OpenAlex by DOI so citation counts are current. A paper is included if it passes the gate, scores at least `CURATE_MIN_SCORE` and has a valid PDF; there's no cap on the library size. PDFs already on disk are reused. Walking best score first, up to `CURATE_TARGET_PDFS` new PDFs are downloaded per run, trying every open-access link.

   `--rejudge` skips the search and downloads: it re-fetches only the manifest papers by DOI, re-scores them with the current rules and rewrites the included flags. Use it after changing the gate, weights or `CURATE_MIN_SCORE`.

   PDFs are saved as `data/papers/<paper_id>.pdf` with readable IDs like `schoenfeld_2017`, and are never deleted. `data/curation_manifest.json` lists every judged paper with `"included": true|false` and, when excluded, a `"reason"` (`failed relevance gate`, `score 0.612 below 0.65`, `no valid PDF (HTTP 403)`, `no PDF on disk (not downloaded)`, `not found in OpenAlex`, …). Papers keep their IDs across runs. The manifest is only replaced when a run finishes; if anything fails, the previous one is kept and the command exits with an error.

   OpenAlex bills each request against a daily budget (about 10 credits per search; without a key your IP shares 1,000 credits a day). A default run uses about 8 searches plus one lookup per 50 manifest papers.

2. **Ingest** — chunk, embed, store in Chroma:
   ```bash
   python -m src.ingest
   ```
   Every run rebuilds Chroma from scratch using only papers the manifest marks as included. It builds into a staging collection and swaps it in at the end, so a failed run leaves the old index working. It prints a warning listing excluded papers and any PDFs in `data/papers/` that aren't in the manifest; neither is ingested. (`--reset` is still accepted but does nothing.)

   Before chunking, front matter (repository cover pages, author lists, everything before an `Abstract` heading) and the reference list are dropped. Tables and figures that come after the references in author manuscripts are kept. If a paper has no `Abstract` heading, nothing is cut from the front.

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

`refused: true` means no excerpt cleared the relevance threshold, so no LLM call was made. `unverified_citations` lists citations the LLM made that aren't retrieved papers: unknown paper IDs and author-style citations like `[Cribb et al.]`.

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
├── .env.example          # OPENALEX_MAILTO, OPENALEX_API_KEY
├── data/
│   ├── papers/           # PDFs (gitignored)
│   ├── chroma/           # vector store (gitignored)
│   └── curation_manifest.json  # every judged paper: metadata, included, reason (gitignored)
├── src/
│   ├── config.py         # paths, models, retrieval knobs
│   ├── curate.py         # OpenAlex search + re-fetch → gate → score → judge → PDFs
│   ├── manifest.py       # metadata bridge between curate and ingest
│   ├── ingest.py         # PaperLoader, split_sections, Chunker, VectorStore
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
| `EMBED_MODEL`      | `nomic-embed-text`                     | Changing it requires re-running `ingest`        |
| `RERANK_MODEL`     | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Any Hugging Face cross-encoder                  |
| `CHUNK_SIZE`       | 800                                    | Characters per chunk                            |
| `CHUNK_OVERLAP`    | 150                                    | Preserves context across splits                 |
| `CANDIDATE_K`      | 20                                     | Vector-search candidates sent to the reranker   |
| `TOP_K`            | 5                                      | Max excerpts passed to the LLM                  |
| `MIN_RERANK_SCORE` | 0.1                                    | 0–1; below this a chunk is dropped              |
| `QUALITY_WEIGHT`   | 0.1                                    | How much curation score breaks relevance ties   |
| `CURATE_QUERIES`   | 8 queries                              | OpenAlex searches; candidates split across them |
| `CURATE_CANDIDATES`| 1000                                   | Total candidates fetched (200 per page)         |
| `CURATE_TARGET_PDFS`| 20                                    | New PDFs downloaded per run (not a library cap) |
| `CURATE_MIN_SCORE` | 0.65                                   | 0–1 quality cutoff before download; `None` = no cut |
| `CURATE_MAX_DOWNLOAD_ATTEMPTS` | 100                        | Safety cap on papers tried                      |
| `CITATION_CAP`     | 50                                     | Citations/year that earns the full citation score |

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
- Only open-access papers are fetched, and only papers in the manifest are ingested. PDFs added by hand are listed as "not in manifest" and skipped.
- Papers are used locally for retrieval and citation only, not redistributed.
- The LLM can still misread excerpts. Citations let users verify.
- Not medical or professional advice.

## License

MIT
