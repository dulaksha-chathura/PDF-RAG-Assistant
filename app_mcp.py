import os
import asyncio
import tempfile
import streamlit as st
from mcp_agent.mcp_server import ingest_pdf_to_database
from mcp_agent.agent import ask_mcp_agent

st.set_page_config(page_title="3-Tier MCP RAG Architecture", layout="wide")
st.title("3-Tier MCP Knowledge Base System")

# Get credentials from environment or Streamlit secrets
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def extract_clean_text(response) -> str:
    """
    Extracts plain text string from raw agent responses (lists, dicts, or objects)
    to prevent rendering raw JSON blobs with signatures.
    """
    if isinstance(response, str):
        return response

    # Handle list response: [{ "type": "text", "text": "...", ... }]
    if isinstance(response, list) and len(response) > 0:
        first_item = response[0]
        if isinstance(first_item, dict):
            return first_item.get("text", str(first_item))
        elif hasattr(first_item, "text"):
            return getattr(first_item, "text")

    # Handle dictionary response: { "output": "..." } or { "text": "..." }
    if isinstance(response, dict):
        if "output" in response:
            return response["output"]
        if "text" in response:
            return response["text"]

    # Handle object with attributes
    if hasattr(response, "content"):
        return response.content
    if hasattr(response, "text"):
        return response.text

    return str(response)


tab1, tab2 = st.tabs(["1. Upload & Index PDF", "2. Ask Questions"])

# -----------------------------------------------------------------------------
# TAB 1: DOCUMENT INGESTION PHASE (Backend Service)
# -----------------------------------------------------------------------------
with tab1:
    st.header("Document Ingestion Service")
    st.write("Upload a PDF file to process text chunks, generate embeddings, and store them in Neon DB.")

    uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"])

    if uploaded_file and st.button("Process & Index PDF"):
        with st.spinner("Parsing document and embedding chunks into Neon DB..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name

            # Ingest file into database
            success = ingest_pdf_to_database(tmp_path, original_filename=uploaded_file.name)
            
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            if success:
                st.success(f"Successfully ingested and indexed **{uploaded_file.name}** into Neon PostgreSQL!")
            else:
                st.error("Failed to process and index the PDF document. Check terminal logs.")

# -----------------------------------------------------------------------------
# TAB 2: QUERY AGENT PHASE (MCP Agent Tool Call)
# -----------------------------------------------------------------------------
with tab2:
    st.header("Query Knowledge Assistant")
    user_input = st.text_input("Ask a question based on indexed documents:")

    if user_input and st.button("Submit Query"):
        if not NEON_DATABASE_URL or not GEMINI_API_KEY:
            st.error("Please ensure NEON_DATABASE_URL and GEMINI_API_KEY environment variables are set.")
        else:
            with st.spinner("Agent is retrieving context via MCP tool..."):
                raw_answer = asyncio.run(
                    ask_mcp_agent(
                        user_query=user_input,
                        neon_db_url=NEON_DATABASE_URL,
                        gemini_api_key=GEMINI_API_KEY,
                    )
                )
                clean_answer = extract_clean_text(raw_answer)

            st.write("### Response:")
            st.markdown(clean_answer)
