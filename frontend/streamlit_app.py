import os

import requests
import streamlit as st

API_URL = os.environ.get("RESUME_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Resume AI", page_icon="📄", layout="centered")

st.title("📄 Resume AI")
st.caption("Ask questions about the resume and get answers pulled straight from it.")

with st.sidebar:
    st.subheader("Settings")
    api_url = st.text_input("API URL", value=API_URL)
    top_k = st.slider("Sources to retrieve", min_value=1, max_value=5, value=3)

    st.divider()
    if st.button("Check API health"):
        try:
            resp = requests.get(f"{api_url}/health", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            st.success(f"API is {data['status']} — {data['chunks_indexed']} chunks indexed.")
        except Exception as e:
            st.error(f"Couldn't reach the API: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['section']}** (score: {s['score']:.2f})")
                    st.write(s["text"])

question = st.chat_input("Ask something about the resume, e.g. 'What AI experience do they have?'")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the resume..."):
            try:
                resp = requests.post(
                    f"{api_url}/api/query",
                    json={"question": question, "top_k": top_k},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                st.markdown(data["answer"])
                st.caption(f"Confidence: {data['confidence']:.2f}")
                if data.get("sources"):
                    with st.expander("Sources"):
                        for s in data["sources"]:
                            st.markdown(f"**{s['section']}** (score: {s['score']:.2f})")
                            st.write(s["text"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                    }
                )
            except Exception as e:
                error_msg = f"Something went wrong talking to the API: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
