import os
import streamlit as st
import psycopg2
from pypdf import PdfReader
from openai import OpenAI

# ----------------------------------------------------
# 1. HARDCODED SYSTEM CREDENTIALS (FAIL-SAFE CONFIG)
# ----------------------------------------------------
# ⚠️ PASTE YOUR ACTUAL NEON POSTGRESQL CONNECTION STRING HERE
NEON_DATABASE_URL = "postgresql://neondb_owner:npg_Jzie01ZjrRBl@ep-snowy-lab-az96t41y-pooler.c-3.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

# ⚠️ PASTE YOUR ACTUAL OPENROUTER API KEY HERE (Starts with sk-or-v1-...)
os.environ["OPENROUTER_API_KEY"] = "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"

# Initialize OpenAI Client mapped directly to OpenRouter's free infrastructure
try:
    api_key_str = os.environ.get("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai",
        api_key=api_key_str,
        default_headers={
            "HTTP-Referer": "http://localhost:8501", 
            "X-Title": "Local RAG Application"
        }
    )
except Exception as e:
    st.error(f"AI Client failed to initialize: {e}")

def get_db_connection():
    """Establishes connection to the Neon serverless PostgreSQL instance."""
    return psycopg2.connect(NEON_DATABASE_URL)

# ----------------------------------------------------
# 2. CORE RAG FUNCTIONS (INGESTION, EMBEDDING, RETRIEVAL)
# ----------------------------------------------------
def process_pdf(file_obj, chunk_size=1000, overlap=200):
    """Extracts raw text from an uploaded PDF file object and partitions it into clean chunks."""
    reader = PdfReader(file_obj)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
            
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embedding(text_chunk):
    """Generates a text embedding vector via OpenRouter's standard embedding model with robust type checks."""
    try:
        response = client.embeddings.create(
            input=[text_chunk],
            model="openai/text-embedding-3-small"
        )
        
        # Defensive check: Handles cases where OpenRouter yields a raw text error string
        if isinstance(response, str):
            raise ValueError(f"OpenRouter returned a text error message instead of an object: {response}")
            
        return response.data[0].embedding if hasattr(response, 'data') and isinstance(response.data, list) else response.data.embedding
        
    except Exception as e:
        raise RuntimeError(f"Embedding generation failed. Check your API key. Details: {e}")

def ingest_pdf_to_db(file_obj):
    """Processes PDF chunks, generates vector profiles, and pushes them straight to Neon Postgres."""
    chunks = process_pdf(file_obj)
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Ensures the table is fresh and optimized for standard 1536 vector element spaces
    cur.execute("DROP TABLE IF EXISTS pdf_documents;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pdf_documents (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding VECTOR(1536)
        );
    """)
    conn.commit()
    
    for chunk in chunks:
        embedding = get_embedding(chunk)
        cur.execute(
            "INSERT INTO pdf_documents (content, embedding) VALUES (%s, %s);",
            (chunk, embedding)
        )
    conn.commit()
    cur.close()
    conn.close()
    return len(chunks)

def query_similar_chunks(user_query, top_k=3):
    """Executes high-speed semantic search using PostgreSQL pgvector cosine distance math (<=>)."""
    query_embedding = get_embedding(user_query)
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute(
        "SELECT content FROM pdf_documents ORDER BY embedding <=> %s::vector LIMIT %s;",
        (query_embedding, top_k)
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [row[0] for row in results]

def ask_rag(user_query):
    """Fuses vector database context with original query and maps response generation via Gemini 2.5 Flash."""
    contexts = query_similar_chunks(user_query)
    combined_context = "\n---\n".join(contexts)
    
    response = client.chat.completions.create(
        model="google/gemini-2.5-flash:free", 
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert AI assistant. Answer using ONLY the provided context info. "
                    "If you do not know the answer based on the context, say 'I cannot find that in the document'."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{combined_context}\n\nQuestion: {user_query}"
            }
        ],
        temperature=0.2
    )
    return response.choices.message.content

# ----------------------------------------------------
# 3. STREAMLIT VISUAL WEB APPLICATION FRONT-END
# ----------------------------------------------------
st.set_page_config(page_title="RAG Document Assistant", layout="centered")
st.title("📄 Gemini 2.5 Flash - PDF RAG Assistant")
st.write("Upload a PDF to store its vector representations in Neon PostgreSQL and chat with it locally.")

# Sidebar tracker mapping active configurations
with st.sidebar:
    st.header("System Monitor")
    if os.environ.get("OPENROUTER_API_KEY") and "your-actual-copied" not in os.environ.get("OPENROUTER_API_KEY"):
        st.success("✅ Gemini (OpenRouter) Key Loaded")
    else:
        st.error("❌ Missing/Placeholder OPENROUTER_API_KEY")

# Drag & Drop upload deck UI
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process & Index Document"):
        with st.spinner("Extracting text and uploading vector embeddings to Neon..."):
            try:
                num_chunks = ingest_pdf_to_db(uploaded_file)
                st.success(f"Successfully partitioned into {num_chunks} chunks and stored in vector database!")
            except Exception as e:
                st.error(f"Error during document ingestion phase: {e}")

st.divider()

# Maintain conversational history blocks within the UI session state parameters
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Multi-turn chat box processing loop
if user_input := st.chat_input("Ask a question about your uploaded document:"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Searching Neon vectors and generating response..."):
            try:
                answer = ask_rag(user_input)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Failed to generate answer. Critical Error: {e}")
