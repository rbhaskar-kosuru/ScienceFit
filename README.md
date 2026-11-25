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
Ingest   included papers → strip line numbers, front matter and references → label sections
         → chunks (never crossing a section) → embed "Paper: title. Section: …" + chunk
         → rebuild Chroma from scratch (metadata and section on every chunk)
Ask      question → under 3 words and no fitness term? refuse, no search
                 → embed → top-100 chunks (Chroma) → max 4 per paper → up to 20 candidates
                 → cross-encoder rerank (header + chunk) → drop chunks below relevance threshold
                 → none left? refuse, no LLM call
                 → top-5 excerpts, max 2 per paper → LLM (temperature 0) → strip reasoning → answer
                 → citation validator → sources + flagged invented citations
```

- **Curation metadata survives ingest.** Title, year, journal, DOI, study design and quality score are stored on every chunk, so citations link to real papers.
- **Section labels.** Every chunk is labelled `abstract`, `introduction`, `methods`, `results`, `discussion`, `table` or `unknown`. Introductions mostly summarise *other* studies, so they are ranked lower and the LLM is told not to present them as the cited paper's own findings.
- **Delimited excerpts.** The LLM sees each chunk as `<excerpt id="grgic_2021" section="discussion">…</excerpt>`, with no title, journal or design inside, and is told to answer in its own words, cite with `[id]` and never reproduce the markup. The user prompt lists the allowed citation ids (the papers in the excerpts); the system prompt has no example paper ID, because the 3B model copied it into answers.
- **Contextual headers.** Each chunk is embedded and reranked as `Paper: <title>. Section: <section>.` + the chunk, so a chunk saying "we found…" still carries its paper and section. Chroma stores the raw chunk.
- **Two-stage retrieval.** Vector search is fast but coarse; the cross-encoder reads question and chunk together and reorders them. Rank score = logit + `QUALITY_WEIGHT` × curation score − `INTRO_PENALTY` (introduction chunks only). Before reranking, Chroma returns the 100 closest chunks and at most `CANDIDATE_PER_PAPER` per paper go on, up to `CANDIDATE_K`: the title in the contextual header makes every chunk of a title-matching paper look close, and without this cap one paper filled all 20 slots. The cross-encoder logit runs about −11 to +11, so quality breaks near-ties without overriding a clearly better match. At most `MAX_CHUNKS_PER_PAPER` chunks per paper reach the LLM, so one paper can't crowd out the rest.
- **Honest refusal.** Short off-topic questions ("hi", "thanks!") are refused before searching. Otherwise, if no chunk's relevance (sigmoid of the logit, 0–1) clears `MIN_RERANK_SCORE`, the API says so instead of guessing.
- **Citation validation.** Every `[paper_id]` and parenthesised `(paper_id)` in the answer is checked against the retrieved papers, case-insensitively (`(Borde_2015)` counts as `borde_2015`; reported IDs are lowercase). Invented IDs and bracketed author-style citations (`[Cribb et al.]`, `[Smith 2019]`) are flagged; brackets that aren't citations (`[P < 0.05]`, `[n = 12]`, `[95% CI …]`) and author-style text in parentheses are ignored.

## Stack

| Layer      | Tool                                                  |
|------------|-------------------------------------------------------|
| LLM        | Ollama (`llama3.2:3b`) by default; any OpenAI-compatible API |
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

# Optional: OpenAlex email ("polite pool") and free API key, hosted LLM settings
cp .env.example .env           # then edit OPENALEX_* and, for a hosted LLM, LLM_*
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

   Study design comes from the first of these that says something: (1) a title that declares itself a narrative review, position stand or consensus statement; (2) PubMed publication types (NCBI E-utilities; PMID from OpenAlex, or looked up by DOI for manifest papers); (3) keywords in the title, then the abstract. Title first, because PubMed tags some narrative reviews `Meta-Analysis` and some position stands only `Review`. Tiers: meta-analysis 1.0, systematic review 0.9, RCT 0.8, position stand 0.6, cohort 0.5, narrative review 0.4, observational 0.4, other 0.3. The manifest keeps `pmid`, `pubtypes` and `design_source` (`pubmed` or `keyword`) and reuses PubMed-based types on later runs. PubMed requests carry `tool` and your `OPENALEX_MAILTO` email; if PubMed fails, the run stops and the previous manifest is kept.

   Every run judges **all** papers together: the new candidates plus every paper already in the manifest, re-fetched from OpenAlex by DOI so citation counts are current. A paper is included if it passes the gate, scores at least `CURATE_MIN_SCORE` and has a valid PDF; there's no cap on the library size. PDFs already on disk are reused. Walking best score first, up to `CURATE_TARGET_PDFS` new PDFs are downloaded per run, trying every open-access link.

   `--rejudge` skips the search and downloads: it re-fetches only the manifest papers by DOI, re-scores them with the current rules and rewrites the included flags. Use it after changing the gate, weights or `CURATE_MIN_SCORE`.

   PDFs are saved as `data/papers/<paper_id>.pdf` with readable IDs like `schoenfeld_2017`, and are never deleted. `data/curation_manifest.json` lists every judged paper with `"included": true|false` and, when excluded, a `"reason"` (`failed relevance gate`, `score 0.612 below 0.65`, `no valid PDF (HTTP 403)`, `no PDF on disk (not downloaded)`, `not found in OpenAlex`, …). Papers keep their IDs across runs. The manifest is only replaced when a run finishes; if anything fails, the previous one is kept and the command exits with an error.

   OpenAlex bills each request against a daily budget (about 10 credits per search; without a key your IP shares 1,000 credits a day). A default run uses about 8 searches plus one lookup per 50 manifest papers.

2. **Ingest** — chunk, embed, store in Chroma:
   ```bash
   python -m src.ingest
   ```
   Every run rebuilds Chroma from scratch using only papers the manifest marks as included. It builds into a staging collection and swaps it in at the end, so a failed run leaves the old index working. It prints a warning listing excluded papers and any PDFs in `data/papers/` that aren't in the manifest; neither is ingested. (`--reset` is still accepted but does nothing.)

   Before chunking, line numbers in author manuscripts are stripped (only when most lines start with a rising number, so numeric table rows survive), then front matter (repository cover pages, author lists, everything before an `Abstract` heading) and the reference list are dropped. Tables and figures that come after the references in author manuscripts are kept. If a paper has no `Abstract` heading, nothing is cut from the front.

   The body is split at headings on their own line (`Introduction`, `2. Methods`, `Results`, `Discussion`, `Conclusions`, …) and chunks never cross a section. Headings are compared without spaces or case, so small-caps ones (`REsulT s`, or one split over several lines) count. An abstract ends at a numbered heading, an `Introduction` heading or the keywords line; before that, `Methods`/`Results`/`Conclusions` lines are a structured abstract. With no `Introduction` heading, the text after the keywords is the introduction. If a section's heading is numbered, unnumbered lines with the same name (table columns) are ignored. Numbered topic sections of narrative reviews (`2 Training Frequency`) end the introduction and count as `discussion`. `table` covers tables, from the caption to the end of the page or until running text resumes, and everything kept after the references. Text before the first heading, or a paper with no headings, is `unknown`. Each paper's section counts are printed.

   Each chunk also stores `secondhand`: true when sentences that cite other studies, and don't mention the paper's own work (`we`, `our`, `the present study`), make up at least half of it. Never set for `table` and `methods` chunks, except that every chunk of a `narrative_review` paper has it. It is stored only; ranking and prompts don't use it yet.

   **Re-ingest after pulling these changes.** New section labels change the embeddings and chunks now carry `secondhand`, so the collection is now `sciencefit_v4` and is empty until you run `python -m src.ingest --reset`. Restart the API afterwards. The old `sciencefit_v3` collection is left in place.

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
     "cited": true, "relevance": 0.94,
     "chunks": [{"index": 12, "section": "results"}, {"index": 4, "section": "introduction"}]}
  ],
  "unverified_citations": [],
  "no_citations": false
}
```

`refused: true` means the question was short and off-topic, or no excerpt cleared the relevance threshold; either way no LLM call was made. `relevance` is the 0–1 sigmoid of the best chunk's reranker logit. `unverified_citations` lists citations the LLM made that aren't retrieved papers: unknown paper IDs and author-style citations like `[Cribb et al.]`. `no_citations: true` means the answer wasn't refused but cites none of its sources; the UI labels each source "✓ cited" or "retrieved, not cited" and shows a warning when `no_citations` is true.

### Smoke eval

`evals/smoke.json` holds nine questions with the papers each should cite (or `must_refuse`). Run them against the live index (Ollama running):

```bash
mkdir -p evals/runs
python -m evals.run_smoke > evals/runs/new.txt     # compare with evals/baseline.txt
python -m evals.run_smoke --model MODEL --base-url URL --prompt-version v1   # override .env for one run
```

The output starts with `LLM_MODEL` and `PROMPT_VERSION`. For each question it reports PASS/FAIL, whether it refused, the retrieved and cited paper IDs, the sections of the cited papers' chunks, any unverified citations, the answer and the excerpts the model saw. Answers cited only from introduction chunks get a WARNING line (background, not the papers' own findings). A question passes if it cites at least one expected paper, or refuses when it must. It does not check that claims are faithful to the excerpts. Generation uses `LLM_TEMPERATURE = 0` and `LLM_SEED`, so reruns should be close but aren't guaranteed identical.

Outputs contain paper excerpts, so keep them in `evals/runs/` (gitignored). The tracked `evals/*.txt` files hold answers only.

#### Results: model × prompt (November 2026)

Auto-check passes out of 9 questions (2 of them must be refused), with identical retrieval:

| Model                 | Prompt v1 | Prompt v2          |
|-----------------------|-----------|--------------------|
| `llama3.2:3b`         | 8/9       | 7/9                |
| `openai/gpt-oss-120b` | 9/9*      | 9/9 (both of 2 runs) |

Answers naming authors in the text: `llama3.2:3b` 6 (v1) → 2 (v2); `openai/gpt-oss-120b` 1 (v1) → 0 (v2, both runs).

\* 4/9 as first run: 5 answers cited with `【id】`, which the validator then ignored. Re-checking the same saved answers with `【id】` read as `[id]` gives 9/9.

- Recommended: `openai/gpt-oss-120b` + v2 for hosted, `llama3.2:3b` + v1 for local (v2 made the 3B model drop citations).
- Hosted runs vary even at temperature 0 with a seed: 3 of 7 generated answers differed between two identical runs.
- The auto-check only tests whether an expected paper is cited, not whether claims are faithful, and v2's rules were written from failures on these same questions.

To tune the threshold without calling the LLM:

```bash
python -m src.rag "does training to failure increase hypertrophy?"
```

It prints each candidate's raw logit, 0–1 relevance, rank score, keep/drop and section.

### Tests

```bash
pytest
```

Ollama and the reranker are faked, and the LLM client talks to a fake local HTTP server, so tests run offline.

## Project structure

```
ScienceFit/
├── app.py                # Streamlit UI
├── requirements.txt
├── .env.example          # OPENALEX_*, LLM_*
├── data/
│   ├── papers/           # PDFs (gitignored)
│   ├── chroma/           # vector store (gitignored)
│   └── curation_manifest.json  # every judged paper: metadata, included, reason (gitignored)
├── src/
│   ├── config.py         # paths, models, retrieval knobs
│   ├── curate.py         # OpenAlex search + re-fetch → gate → score → judge → PDFs
│   ├── manifest.py       # metadata bridge between curate and ingest
│   ├── ingest.py         # PaperLoader, line numbers, sections, Chunker, contextual headers, VectorStore
│   ├── rerank.py         # cross-encoder reranker (raw logits)
│   ├── llm.py            # OpenAI-compatible client: retries, reasoning cleanup
│   ├── prompts.py        # versioned prompts (v1 to v4)
│   ├── rag.py            # retrieve → rerank → refuse/generate → validate
│   └── api.py            # FastAPI endpoints
├── evals/
│   ├── smoke.json        # smoke questions + expected papers
│   ├── run_smoke.py      # runs them through RAG().ask()
│   ├── *.txt             # tracked outputs, answers only
│   └── runs/             # outputs with paper excerpts (gitignored)
└── tests/
    ├── conftest.py       # blanks LLM_* so tests ignore .env
    ├── test_design.py    # study design: titles, PubMed types, keywords
    ├── test_llm.py
    ├── test_sections.py  # section labels, second-hand flag
    └── test_pipeline.py
```

## Configuration

Tune in `src/config.py` (the `LLM_*` settings marked `.env` are read from there):

| Setting            | Default                                | Notes                                           |
|--------------------|----------------------------------------|-------------------------------------------------|
| `LLM_BASE_URL`     | `http://localhost:11434/v1`            | `.env`; any OpenAI-compatible API               |
| `LLM_MODEL`        | `llama3.2:3b`                          | `.env`; no re-ingest needed                     |
| `LLM_PROMPT_VERSION` | `v1`                                 | `.env`; `v1` to `v4` (see Swapping the LLM)     |
| `LLM_API_KEY`      | empty                                  | `.env`; not needed for Ollama                   |
| `LLM_EXTRA_PARAMS` | empty                                  | `.env`; JSON of provider-only request fields    |
| `LLM_TIMEOUT`      | 180                                    | Seconds per LLM request                         |
| `LLM_MAX_RETRIES`  | 5                                      | Retries on HTTP 429/5xx, with backoff           |
| `LLM_MAX_RETRY_WAIT` | 60                                   | Seconds; a longer `Retry-After` fails at once   |
| `LLM_TEMPERATURE`  | 0                                      | Deterministic decoding for repeatable evals     |
| `LLM_SEED`         | 42                                     | Sampling seed                                   |
| `EMBED_MODEL`      | `nomic-embed-text`                     | Changing it requires re-running `ingest`        |
| `RERANK_MODEL`     | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Any Hugging Face cross-encoder                  |
| `COLLECTION`       | `sciencefit_v4`                        | Bumped when embeddings change; re-ingest after  |
| `CHUNK_SIZE`       | 800                                    | Characters per chunk; changing it needs a re-ingest |
| `CHUNK_OVERLAP`    | 150                                    | Preserves context across splits                 |
| `CANDIDATE_POOL`   | 100                                    | Chunks pulled from Chroma before the per-paper cap |
| `CANDIDATE_PER_PAPER` | 4                                   | Max chunks per paper among reranker candidates  |
| `CANDIDATE_K`      | 20                                     | Vector-search candidates sent to the reranker   |
| `TOP_K`            | 5                                      | Max excerpts passed to the LLM                  |
| `MIN_RERANK_SCORE` | 0.1                                    | 0–1 (sigmoid of logit); below this a chunk is dropped |
| `QUALITY_WEIGHT`   | 0.5                                    | Logit bonus × curation score (0–1); breaks near-ties |
| `INTRO_PENALTY`    | 2.0                                    | Logits subtracted from introduction chunks      |
| `MAX_CHUNKS_PER_PAPER` | 2                                  | Cap per paper among the `TOP_K` excerpts        |
| `FITNESS_TERMS`    | 28 words                               | Questions under 3 words need one of these (plurals ok) |
| `CURATE_QUERIES`   | 8 queries                              | OpenAlex searches; candidates split across them |
| `CURATE_CANDIDATES`| 1000                                   | Total candidates fetched (200 per page)         |
| `CURATE_TARGET_PDFS`| 20                                    | New PDFs downloaded per run (not a library cap) |
| `CURATE_MIN_SCORE` | 0.65                                   | 0–1 quality cutoff before download; `None` = no cut |
| `CURATE_MAX_DOWNLOAD_ATTEMPTS` | 100                        | Safety cap on papers tried                      |
| `CITATION_CAP`     | 50                                     | Citations/year that earns the full citation score |

## Swapping the LLM

Any OpenAI-compatible API works (Ollama, Groq, OpenRouter, Together, …). Set three variables in `.env`, plus the prompt version that suits the model, and restart the API:

```bash
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
LLM_API_KEY=your-key
LLM_PROMPT_VERSION=v2
```

Leave them blank for local Ollama with prompt v1. `LLM_EXTRA_PARAMS` takes provider-only request fields as JSON, e.g. `{"reasoning_effort": "low"}`. With a hosted LLM, the question and the retrieved excerpts are sent to that provider. Embeddings always use Ollama. Retrieval, reranking and validation stay the same.

`LLM_API_KEY` is never sent to `localhost`, `127.0.0.1` or `::1`. `<think>…</think>` text is stripped, any separate reasoning field is ignored and `【id】` citations become `[id]` before citations are validated. Prompts live in `src/prompts.py`: `v1` (default), `v2`, `v3` and `v4`, set with `LLM_PROMPT_VERSION`; the smoke eval can override it with `--prompt-version`.

`v3` is `v2` plus: a one-sentence direct answer first, then plain-language evidence, no unexplained numbers or technical statistics, 3–5 sentences.

`v4` is `v3` revised for accuracy; it asks for a cited, supported opening sentence, comparison with the usual alternative, rounding but no calculating, no lab units, practical limits included, and second-hand findings marked "as reviewed in".

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
