# ScienceFit

RAG chatbot answering resistance-training questions from peer-reviewed research with citations.

## Stack

- **LLM & embeddings**: Ollama (`llama3.1`, `nomic-embed-text`)
- **Vector DB**: Chroma (local, persistent)
- **Backend**: FastAPI
- **Frontend**: Streamlit

## Setup (MacBook Pro M3)

```bash
# 1. Install Ollama, then pull models
brew install ollama
ollama serve &
ollama pull llama3.1
ollama pull nomic-embed-text

# 2. Python env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Drop your ~20 curated PDFs into data/papers/

# 4. Ingest
python -m src.ingest

# 5. Run API (terminal 1)
uvicorn src.api:app --reload

# 6. Run UI (terminal 2)
streamlit run app.py
```

## Project layout

```
sciencefit/
├── app.py              # Streamlit UI
├── requirements.txt
├── data/
│   ├── papers/         # your PDFs go here
│   └── chroma/         # auto-created vector store
└── src/
    ├── config.py       # paths, model names, chunk sizes
    ├── ingest.py       # PDF → chunks → embeddings → Chroma
    ├── rag.py          # retrieve + generate with citations
    └── api.py          # FastAPI endpoints
```

## Next steps

1. Get MVP working end-to-end with 20 papers.
2. Add conversational memory (store chat history, pass previous turns).
3. Build scraper for PubMed / Google Scholar with quality gate.
4. Dockerize.
5. Deploy backend on Railway, frontend on Streamlit Cloud or Vercel.
6. (Optional) Swap Ollama → Claude API by changing `_generate` in `rag.py`.
