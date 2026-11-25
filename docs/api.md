# API

Start it with `uvicorn src.api:app` (add `--reload` while developing). Restart it after a re-ingest: the index is swapped, and a running API keeps a handle to the old one and returns 500.

## GET /health

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "ollama": true, "chunks": 2218, "llm": "llama3.2:3b", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
```

`status` is `ok` when Ollama answers and the index isn't empty, else `degraded`. Ollama is needed for embeddings even with a hosted LLM. The hosted LLM itself is not checked.

## POST /ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "optimal reps for hypertrophy?", "top_k": 5}'
```

| Request field | Type | Notes |
|---|---|---|
| `question` | string | Required, not empty |
| `top_k` | int, 1 to 10 | Most passages sent to the LLM; default `TOP_K` (5) |

Invalid requests get 422. If the LLM fails after its retries, the response is 500.

```json
{
  "answer": "… [schoenfeld_2017] …",
  "refused": false,
  "sources": [
    {"paper_id": "schoenfeld_2017", "title": "…", "first_author": "…", "year": 2017, "journal": "…",
     "doi": "https://doi.org/…", "design": "meta_analysis", "curated": true,
     "cited": true, "relevance": 0.94,
     "chunks": [{"index": 12, "section": "results"}, {"index": 4, "section": "introduction"}]}
  ],
  "unverified_citations": [],
  "no_citations": false
}
```

(The paper above is made up for the example.)

| Response field | Meaning |
|---|---|
| `answer` | The answer, or the no-evidence message. `【id】` citations are returned as `[id]` |
| `refused` | True if the question was short and off-topic, or no passage cleared the relevance threshold. Either way no LLM call was made |
| `sources` | One entry per paper sent to the LLM, cited papers first. Empty when refused. `title`, `first_author`, `year`, `journal`, `doi` and `design` are left out when unknown |
| `sources[].cited` | Whether the answer cites this paper |
| `sources[].relevance` | 0 to 1: sigmoid of the reranker logit of the paper's best passage |
| `sources[].chunks` | Index and section of each passage sent |
| `unverified_citations` | Citations that aren't retrieved papers: unknown paper IDs and author-style ones like `[Cribb et al.]` |
| `no_citations` | True if the answer wasn't refused but cites none of its sources |

Citations are `[paper_id]` or `(paper_id)`, matched case-insensitively (`(Borde_2015)` counts as `borde_2015`) and reported in lowercase; an escaped `[borde\_2015]` counts too. Brackets that aren't citations (`[P < 0.05]`, `[n = 12]`, `[95% CI …]`) and author-style text in parentheses are ignored.

The UI labels each source "✓ cited" or "retrieved, not cited", and warns when `no_citations` is true or `unverified_citations` isn't empty.
