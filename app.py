import streamlit as st
from google import genai

from services.database import initialize_database, search_similar_chunks, insert_document_chunks
from services.embeddings import get_embedding, vector_to_pgvector
from services.pdf_processor import process_pdf
from ui.sidebar import render_sidebar
from ui.chat import init_chat_history, render_chat_history, add_message

# Streamlit Page Setup
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="centered")

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
st.title("📄 Native Gemini - PDF RAG Assistant")
st.write("Upload a PDF, store its embeddings in Neon PostgreSQL, and ask questions about it.")

# PDF Ingestion Pipeline
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

# RAG Generation Function
def ask_native_rag(user_query: str) -> str:
    query_vector = get_embedding(gemini_client, user_query, task_type="RETRIEVAL_QUERY")
    vector_str = vector_to_pgvector(query_vector)
    contexts = search_similar_chunks(NEON_DATABASE_URL, vector_str, top_k=5)
    
    if not contexts:
        return "I cannot find relevant information in the document."
    
    combined_context = "\n\n---\n\n".join(contexts)
    prompt = f"""
    You are an expert document assistant.
    Answer the user's question accurately using ONLY the information provided in the Document Context below.
    If details are missing, state clearly what is missing.

    Document Context:
    {combined_context}

    Question:
    {user_query}
    """
    
    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text

# Process User Chat Input
user_input = st.chat_input("Ask a question about your uploaded document...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    add_message("user", user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            try:
                answer = ask_native_rag(user_input)
                st.markdown(answer)
                add_message("assistant", answer)
            except Exception as e:
                st.error("❌ Failed to generate answer.")
                st.exception(e)
