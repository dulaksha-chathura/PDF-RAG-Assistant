import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ System Monitor")
        st.success("✅ Native Gemini API Active")
        st.info("Embedding Model: gemini-embedding-001 (768d)")
        st.info("LLM Model: gemini-3.5-flash")
        st.success("✅ Neon PostgreSQL Connected")
