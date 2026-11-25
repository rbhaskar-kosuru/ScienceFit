# Configuration

Settings live in `src/config.py`; those marked `.env` are read from `.env` (copy `.env.example`). Blank or missing `.env` values use the defaults.

## Settings

| Setting            | Default                                | Notes                                           |
|--------------------|----------------------------------------|-------------------------------------------------|
| `LLM_BASE_URL`     | `http://localhost:11434/v1`            | `.env`; any OpenAI-compatible API               |
| `LLM_MODEL`        | `llama3.2:3b`                          | `.env`; no re-ingest needed                     |
| `LLM_PROMPT_VERSION` | `v1`                                 | `.env`; `v1` to `v5` (see LLM and prompts)      |
| `LLM_API_KEY`      | empty                                  | `.env`; not needed for Ollama                   |
| `LLM_EXTRA_PARAMS` | empty                                  | `.env`; JSON of provider-only request fields    |
| `LLM_TIMEOUT`      | 180                                    | Seconds per LLM request                         |
| `LLM_MAX_RETRIES`  | 5                                      | Retries on HTTP 429/5xx, with backoff           |
| `LLM_MAX_RETRY_WAIT` | 60                                   | Seconds; a longer `Retry-After` fails at once   |
| `LLM_TEMPERATURE`  | 0                                      | Kept at 0 so eval runs are comparable     |
| `LLM_SEED`         | 42                                     | Sampling seed                                   |
| `OPENALEX_MAILTO`  | empty                                  | `.env`; contact email sent to OpenAlex and PubMed |
| `OPENALEX_API_KEY` | empty                                  | `.env`; optional, sent as a header              |
| `EMBED_MODEL`      | `nomic-embed-text`                     | Changing it requires re-running `ingest`        |
| `RERANK_MODEL`     | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Any Hugging Face cross-encoder                  |
| `DOC_PREFIX`, `QUERY_PREFIX` | `search_document: `, `search_query: ` | Expected by the embedder; changing them needs a re-ingest |
| `EMBED_BATCH`      | 32                                     | Chunks embedded per request                     |
| `COLLECTION`       | `sciencefit_v4`                        | Bumped when the chunk schema or embeddings change; re-ingest after  |
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
| `CURATE_FROM_YEAR` | 2005                                   | Oldest publication year searched                |
| `CURATE_MIN_SCORE` | 0.65                                   | 0–1 quality cutoff before download; `None` = no cut |
| `CURATE_MAX_DOWNLOAD_ATTEMPTS` | 100                        | Safety cap on papers tried                      |
| `CITATION_CAP`     | 50                                     | Citations/year that earns the full citation score |

## What needs a re-ingest

- **No re-ingest:** LLM, prompt version, reranker, retrieval settings (`CANDIDATE_*`, `TOP_K`, `MIN_RERANK_SCORE`, `QUALITY_WEIGHT`, `INTRO_PENALTY`, `MAX_CHUNKS_PER_PAPER`). Restart the API.
- **Re-ingest:** `EMBED_MODEL`, the prefixes, `CHUNK_SIZE`, `CHUNK_OVERLAP`, section rules, or anything that changes the manifest (included papers, scores, designs). Run `python -m src.ingest`, then restart the API.
- **Curation rules** (gate, weights, `CURATE_MIN_SCORE`): `python -m src.curate --rejudge`, then re-ingest.
- **After pulling a version with a new `COLLECTION`:** the collection is empty until you re-ingest. The manifest and index are not in git.

## LLM and prompts

Any OpenAI-compatible chat-completions API works (Ollama, Groq, OpenRouter, Together, …):

```bash
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
LLM_API_KEY=gsk_...
LLM_PROMPT_VERSION=v2
```

Leave them blank for local Ollama with prompt v1. `LLM_EXTRA_PARAMS` takes provider-only request fields as JSON, e.g. `{"reasoning_effort": "low"}`. With a hosted LLM, the question and the retrieved passages are sent to that provider. Embeddings always use Ollama. `<think>…</think>` text is stripped, any separate reasoning field is ignored and `【id】` citations become `[id]` before citations are validated.

| Prompt | What it is |
|---|---|
| `v1` | Default |
| `v2` | Numbered rules: allowed IDs only, no author names, numbers as written, introductions are background |
| `v3` | v2 plus a one-sentence direct answer first, plain language, no unexplained numbers or technical statistics, 3 to 5 sentences |
| `v4` | v3 revised for accuracy: cited and supported opening sentence, comparison with the usual alternative, rounding but no calculating, no lab units, practical limits, second-hand findings marked "as reviewed in" |
| `v5` | v2 plus one rule: passages tagged `secondhand="true"` (only v5 shows the tag) are cited "as reviewed in [id]", never as the paper's own finding |

The LLM sees each chunk as `<excerpt id="grgic_2021" section="discussion">…</excerpt>`, with no title, journal or design inside, and is told to cite with `[id]` and never reproduce the markup. The user prompt lists the allowed IDs. The system prompts contain no example paper ID, because the 3B model copied it into answers.

## API keys

Keys live only in `.env` (gitignored) and are sent in `Authorization` headers, never in URLs. `LLM_API_KEY` is never sent to `localhost`, `127.0.0.1` or `::1`. Errors and logs name the service and the status code or error type, never the request URL or headers; `src/redact.py` strips keys from error text that could hold one. `tests/test_secrets.py` forces failures in every client with fake keys, checks messages, output and logs, and scans the repo for key-shaped strings. Tests blank the real keys and block the network.

## Curate

```bash
python -m src.curate                                    # CURATE_QUERIES, ~1,000 candidates, up to 20 new PDFs
python -m src.curate --no-download                      # dry run: judge everything, no downloads, manifest untouched
python -m src.curate --rejudge                          # no search or downloads: re-score manifest papers, update flags
python -m src.curate --rejudge --no-download            # dry run of the same
python -m src.curate "training to failure" --target 10  # single custom query
python -m src.curate --candidates 200                   # fewer candidates
```

Candidates are pulled 200 per page across `CURATE_QUERIES`, deduplicated by DOI and limited to open access. Articles and reviews (where OpenAlex files most meta-analyses) are fetched. A paper passes the relevance gate if it mentions a resistance-specific term (resistance, strength, hypertrophy, 1RM, …; `training` alone isn't enough) and no clinical or bench-science term. Matching is whole-word, so `strength` doesn't match `strengthen` and `randomized` doesn't match `non-randomized`. Score = 0.40 × citations (log scale, full marks at `CITATION_CAP`/year) + 0.35 × study design + 0.25 × trusted journal. Papers below `CURATE_MIN_SCORE` are cut.

Study design comes from the first of these that says something: (1) a title that declares itself a narrative review, position stand or consensus statement; (2) PubMed publication types (NCBI E-utilities; PMID from OpenAlex, or looked up by DOI for manifest papers); (3) keywords in the title, then the abstract. Title first, because PubMed tags some narrative reviews `Meta-Analysis` and some position stands only `Review`. Tiers: meta-analysis 1.0, systematic review 0.9, RCT 0.8, position stand 0.6, cohort 0.5, narrative review 0.4, observational 0.4, other 0.3. The manifest keeps `pmid`, `pubtypes` and `design_source` (`pubmed` or `keyword`) and reuses PubMed-based types on later runs. PubMed requests carry `tool` and your `OPENALEX_MAILTO` email; if PubMed fails, the run stops and the previous manifest is kept.

Every run judges **all** papers together: the new candidates plus every paper already in the manifest, re-fetched from OpenAlex by DOI so citation counts are current. A paper is included if it passes the gate, scores at least `CURATE_MIN_SCORE` and has a valid PDF; there's no cap on the library size. PDFs already on disk are reused. Walking best score first, up to `CURATE_TARGET_PDFS` new PDFs are downloaded per run, trying every open-access link.

`--rejudge` skips the search and downloads: it re-fetches only the manifest papers by DOI, re-scores them with the current rules and rewrites the included flags. Use it after changing the gate, weights or `CURATE_MIN_SCORE`.

PDFs are saved as `data/papers/<paper_id>.pdf` with readable IDs like `schoenfeld_2017`, and are never deleted. `data/curation_manifest.json` lists every judged paper with `"included": true|false` and, when excluded, a `"reason"` (`failed relevance gate`, `score 0.612 below 0.65`, `no valid PDF (HTTP 403)`, `no PDF on disk (not downloaded)`, `not found in OpenAlex`, …). Papers keep their IDs across runs. The manifest is only replaced when a run finishes; if anything fails, the previous one is kept and the command exits with an error.

OpenAlex limits usage per day; a free key (openalex.org/settings/api) gives you your own allowance. A default run makes about 8 searches plus one lookup per 50 manifest papers, and calls PubMed for papers whose types aren't cached.

## Ingest

```bash
python -m src.ingest
```

Every run rebuilds Chroma from scratch using only papers the manifest marks as included. It builds into a staging collection and swaps it in at the end, so a failed run leaves the old index working. It prints a warning listing excluded papers and any PDFs in `data/papers/` that aren't in the manifest; neither is ingested. (`--reset` is still accepted but does nothing.)

Before chunking, line numbers in author manuscripts are stripped (only when most lines start with a rising number, so numeric table rows survive), then front matter (repository cover pages, author lists, everything before an `Abstract` heading) and the reference list are dropped. Tables and figures that come after the references in author manuscripts are kept. If a paper has no `Abstract` heading, nothing is cut from the front.

The body is split at headings on their own line (`Introduction`, `2. Methods`, `Results`, `Discussion`, `Conclusions`, …) and chunks never cross a section. Headings are compared without spaces or case, so small-caps ones (`REsulT s`, or one split over several lines) count. An abstract ends at a numbered heading, an `Introduction` heading or the keywords line; before that, `Methods`/`Results`/`Conclusions` lines are a structured abstract. With no `Introduction` heading, the text after the keywords is the introduction. If a section's heading is numbered, unnumbered lines with the same name (table columns) are ignored. Numbered topic sections of narrative reviews (`2 Training Frequency`) end the introduction and count as `discussion`. `table` covers tables, from the caption to the end of the page or until running text resumes, and everything kept after the references. Text before the first heading, or a paper with no headings, is `unknown`. Each paper's section counts are printed.

Each chunk also stores `secondhand`: true when sentences that cite other studies, and don't mention the paper's own work (`we`, `our`, `the present study`), make up at least half of it. Never set for `table` and `methods` chunks, except that every chunk of a `narrative_review` paper has it. Prompt `v5` shows it to the model; ranking and the other prompts don't use it.

Only papers in the manifest are ingested. PDFs added by hand are listed as "not in manifest" and skipped.

## Retrieval

Title, first author, year, journal, DOI, study design, quality score and citations per year are stored on every chunk.

- **Section labels.** Every chunk is labelled `abstract`, `introduction`, `methods`, `results`, `discussion`, `table` or `unknown`. Introductions mostly summarise *other* studies, so they are ranked lower and the LLM is told not to present them as the cited paper's own findings.
- **Delimited excerpts.** The LLM sees each chunk as `<excerpt id="grgic_2021" section="discussion">…</excerpt>`, with no title, journal or design inside, and is told to answer in its own words, cite with `[id]` and never reproduce the markup. The user prompt lists the allowed citation ids (the papers in the excerpts); the system prompt has no example paper ID, because the 3B model copied it into answers.
- **Contextual headers.** Each chunk is embedded and reranked as `Paper: <title>. Section: <section>.` + the chunk, so a chunk saying "we found…" still carries its paper and section. Chroma stores the raw chunk.
- **Two-stage retrieval.** Vector search is fast but coarse; the cross-encoder reads question and chunk together and reorders them. Rank score = logit + `QUALITY_WEIGHT` × curation score − `INTRO_PENALTY` (introduction chunks only). Before reranking, Chroma returns the 100 closest chunks and at most `CANDIDATE_PER_PAPER` per paper go on, up to `CANDIDATE_K`: the title in the contextual header makes every chunk of a title-matching paper look close, and without this cap one paper filled all 20 slots. The cross-encoder logit runs about −11 to +11, so quality breaks near-ties without overriding a clearly better match. At most `MAX_CHUNKS_PER_PAPER` chunks per paper reach the LLM, so one paper can't crowd out the rest.
- **Refusal.** Questions under 3 words with no fitness term ("hi", "thanks!") are refused before searching. Otherwise, if no chunk's relevance (sigmoid of the logit, 0 to 1) reaches `MIN_RERANK_SCORE`, the API refuses without calling the LLM.
- **Chroma's search is approximate.** On the current index it misses 1 to 3 of the 100 closest chunks on some questions, so a rebuild can change retrieval slightly.

To tune the threshold without calling the LLM:

```bash
python -m src.rag "does training to failure increase hypertrophy?"
```

It prints each candidate's raw logit, 0–1 relevance, rank score, keep/drop and section.

## Smoke eval

`evals/smoke.json` holds nine questions with the papers each should cite (or `must_refuse`). It runs against the live index, with Ollama running:

```bash
mkdir -p evals/runs
python -m evals.run_smoke > evals/runs/new.txt
python -m evals.run_smoke --model MODEL --base-url URL --prompt-version v1   # override .env for one run
```

The output starts with `LLM_MODEL` and `PROMPT_VERSION` and ends with how many questions were answered (a question that errors or comes back empty is flagged, and the command exits with an error). For each question it reports PASS/FAIL, whether it refused, the retrieved and cited paper IDs, the sections of the cited papers' chunks, any unverified citations, the answer and the excerpts the model saw. Answers cited only from introduction chunks get a WARNING line (background, not the papers' own findings). A question passes if it cites at least one expected paper, or refuses when it must. It does not check that claims are faithful to the excerpts. Generation uses `LLM_TEMPERATURE = 0` and `LLM_SEED`, so reruns should be close but aren't guaranteed identical.

Outputs contain paper excerpts, so keep them in `evals/runs/` (gitignored). The tracked `evals/*.txt` files hold answers only.

Auto-check results, 9 questions, identical retrieval (index before the section-label fix):

| Model | v1 | v2 |
|---|---|---|
| `llama3.2:3b` | 8/9 | 7/9 |
| `openai/gpt-oss-120b` | 9/9 (4/9 before `【id】` was normalised; same answers re-checked) | 9/9, both of 2 runs |

Answers naming authors in the text: `llama3.2:3b` 6 (v1) → 2 (v2); `openai/gpt-oss-120b` 1 (v1) → 0 (v2, both runs). Hosted runs vary: 3 of 7 generated answers differed between two v2 runs. v2's rules were written from failures on these same questions.

## Tests

`pytest`. Ollama, the reranker, OpenAlex and PubMed are faked, and the LLM client talks to a fake local HTTP server, so tests run offline.

## Stack

Ollama (`llama3.2:3b`, `nomic-embed-text`), `cross-encoder/ms-marco-MiniLM-L-6-v2` via Transformers, Chroma (persistent, local, cosine), FastAPI, Streamlit, pypdf.
