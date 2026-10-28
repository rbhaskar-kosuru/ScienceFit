import requests
import streamlit as st

API_URL = st.sidebar.text_input("API URL", "http://localhost:8000/ask")
TOP_K = st.sidebar.slider("Max excerpts sent to LLM", 1, 10, 5)

st.title("ScienceFit")
st.caption("Research-backed answers on resistance training.")


def ask(question: str) -> dict:
    try:
        r = requests.post(API_URL, json={"question": question, "top_k": TOP_K}, timeout=180)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"answer": f"API error: {e}", "refused": True, "sources": [], "unverified_citations": []}


def source_line(s: dict) -> str:
    title = s.get("title", s["paper_id"])
    link = f"[{title}]({s['doi']})" if s.get("doi") else title
    meta = " · ".join(str(x) for x in (s.get("year"), s.get("journal"), s.get("design")) if x)
    note = "" if s["cited"] else " · _retrieved, not cited_"
    return f"- **[{s['paper_id']}]** {link}  \n  {meta} · relevance {s['relevance']:.2f}{note}"


def render_sources(res: dict) -> None:
    if res.get("unverified_citations"):
        st.warning("Citations not found in the retrieved papers: " + ", ".join(res["unverified_citations"]))
    if res.get("sources"):
        with st.expander(f"Sources ({len(res['sources'])})"):
            st.markdown("\n".join(source_line(s) for s in res["sources"]))


if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            render_sources(turn["res"])

if question := st.chat_input("Ask about hypertrophy, volume, frequency..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reranking papers..."):
            res = ask(question)
        st.markdown(res["answer"])
        render_sources(res)
    st.session_state.history.append({"role": "assistant", "content": res["answer"], "res": res})
