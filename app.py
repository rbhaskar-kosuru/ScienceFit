import streamlit as st
import requests

API_URL = st.sidebar.text_input("API URL", "http://localhost:8000/ask")
TOP_K = st.sidebar.slider("Top K", 1, 10, 5)

st.title("ScienceFit")
st.caption("Research-backed answers on resistance training.")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("citations"):
            with st.expander("Citations"):
                for c in turn["citations"]:
                    st.write(f"- **{c['paper']}** (chunks: {c['chunks']})")

if question := st.chat_input("Ask about hypertrophy, volume, frequency..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving papers..."):
            res = requests.post(API_URL, json={"question": question, "top_k": TOP_K}).json()
        st.markdown(res["answer"])
        with st.expander("Citations"):
            for c in res["citations"]:
                st.write(f"- **{c['paper']}** (chunks: {c['chunks']})")
    st.session_state.history.append({
        "role": "assistant",
        "content": res["answer"],
        "citations": res["citations"],
    })
