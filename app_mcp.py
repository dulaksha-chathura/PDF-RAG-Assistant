import sys
import asyncio
import streamlit as st
from google import genai

from services.database import initialize_database, insert_document_chunks
from services.embeddings import get_embedding, vector_to_pgvector
from services.pdf_processor import process_pdf
from ui.sidebar import render_sidebar
from ui.chat import init_chat_history, render_chat_history, add_message

# Import the Agent execution helper from your mcp_agent package
from mcp_agent.agent import ask_mcp_agent

# Streamlit Page Setup
st.set_page_config(page_title="MCP Agent - PDF RAG Assistant", page_icon="🤖", layout="centered")

# Initialize Credentials
try:
    NEON_DATABASE_URL = st.secrets["NEON_DATABASE_URL"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error("❌ Could not load secrets or initialize Gemini client.")
    st.exception(e)
    st.stop()

# Initialize Database Schema
try:
    initialize_database(NEON_DATABASE_URL)
except Exception as e:
    st.error("❌ Database initialization failed.")
    st.exception(e)
    st.stop()

# Render UI Elements
render_sidebar()
st.title("🤖 Agentic Mode (MCP + LangChain)")
st.caption("Powered by FastMCP, LangChain ReAct Agent, and Google Gemini")

# PDF Ingestion Pipeline (Shared Services Tier)
uploaded_file = st.file_uploader("📂 Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.write(f"Selected file: **{uploaded_file.name}**")
    if st.button("🚀 Process & Index Document"):
        with st.spinner("Extracting PDF and generating Gemini embeddings..."):
            try:
                chunks = process_pdf(uploaded_file)
                data_to_insert = []
                progress = st.progress(0)
                
                for idx, chunk in enumerate(chunks):
                    vector = get_embedding(gemini_client, chunk, task_type="RETRIEVAL_DOCUMENT")
                    data_to_insert.append((chunk, vector_to_pgvector(vector)))
                    progress.progress((idx + 1) / len(chunks))
                
                insert_document_chunks(NEON_DATABASE_URL, data_to_insert)
                progress.empty()
                st.success(f"✅ Indexed {len(chunks)} chunks into Neon PostgreSQL!")
            except Exception as e:
                st.error("❌ Document ingestion failed.")
                st.exception(e)

# Chat UI Setup
init_chat_history()
render_chat_history()

# Process User Chat Input
user_input = st.chat_input("Ask the MCP Agent a question about your document...")

if user_input:
    # Display User Input
    with st.chat_message("user"):
        st.markdown(user_input)
    add_message("user", user_input)

    # Process Assistant Response via FastMCP / LangChain Agent
    with st.chat_message("assistant"):
        with st.spinner("Agent thinking and executing MCP tool calls..."):
            try:
                # Execute the asynchronous agent loop
                answer = asyncio.run(ask_mcp_agent(
                    user_query=user_input,
                    neon_db_url=NEON_DATABASE_URL,
                    gemini_api_key=GEMINI_API_KEY
                ))
                st.markdown(answer)
                add_message("assistant", answer)
            except Exception as e:
                st.error("❌ Agent execution failed.")
                st.exception(e)
