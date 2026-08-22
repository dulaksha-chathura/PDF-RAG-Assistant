import streamlit as st
import psycopg2
from pypdf import PdfReader
from google import genai
from google.genai import types


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="centered"
)


# ============================================================
# 2. LOAD SECRETS & INITIALIZE GEMINI CLIENT
# ============================================================

try:
    NEON_DATABASE_URL = st.secrets["NEON_DATABASE_URL"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error("❌ Could not load secrets or initialize Gemini client.")
    st.error(str(e))
    st.info(
        """
        Make sure you have `.streamlit/secrets.toml` containing:
        
        NEON_DATABASE_URL = "your_neon_connection_string"
        GEMINI_API_KEY = "your_gemini_api_key"
        """
    )
    st.stop()


# ============================================================
# 3. DATABASE CONNECTION & INITIALIZATION
# ============================================================

def get_db_connection():
    return psycopg2.connect(NEON_DATABASE_URL)


@st.cache_resource
def initialize_database():
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pdf_documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding VECTOR(768)
            );
            """
        )
        conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


try:
    initialize_database()
except Exception as e:
    st.error("❌ Database initialization failed.")
    st.exception(e)
    st.stop()


# ============================================================
# 4. PDF TEXT EXTRACTION
# ============================================================

def process_pdf(file_obj, chunk_size=1000, overlap=200):
    reader = PdfReader(file_obj)
    full_text = ""

    for page in reader.pages:
        try:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        except Exception:
            continue

    full_text = " ".join(full_text.split())

    if not full_text.strip():
        raise ValueError("No readable text was found in the PDF.")

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


# ============================================================
# 5. GEMINI EMBEDDING GENERATION
# ============================================================

def get_embedding(text_chunk, task_type="RETRIEVAL_DOCUMENT"):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text_chunk,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768
        )
    )
    return response.embeddings[0].values


def vector_to_pgvector(vector):
    return "[" + ",".join(map(str, vector)) + "]"


# ============================================================
# 6. INGEST PDF INTO NEON
# ============================================================

def ingest_pdf_to_db(file_obj):
    chunks = process_pdf(file_obj)
    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("DROP TABLE IF EXISTS pdf_documents;")
        cur.execute(
            """
            CREATE TABLE pdf_documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding VECTOR(768)
            );
            """
        )
        conn.commit()

        progress = st.progress(0)
        total_chunks = len(chunks)

        for index, chunk in enumerate(chunks):
            embedding = get_embedding(chunk, task_type="RETRIEVAL_DOCUMENT")
            embedding_string = vector_to_pgvector(embedding)

            cur.execute(
                """
                INSERT INTO pdf_documents (content, embedding)
                VALUES (%s, %s::vector);
                """,
                (chunk, embedding_string)
            )
            progress.progress((index + 1) / total_chunks)

        conn.commit()
        progress.empty()
        return total_chunks

    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# ============================================================
# 7. SEMANTIC SEARCH & GEMINI 3.5 FLASH RAG
# ============================================================

def query_similar_chunks(user_query, top_k=5):
    query_embedding = get_embedding(user_query, task_type="RETRIEVAL_QUERY")
    embedding_string = vector_to_pgvector(query_embedding)

    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT content
            FROM pdf_documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (embedding_string, top_k)
        )

        results = cur.fetchall()
        return [row[0] for row in results]
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def ask_rag(user_query):
    contexts = query_similar_chunks(user_query, top_k=5)

    if not contexts:
        return "I cannot find that in the document."

    combined_context = "\n\n---\n\n".join(contexts)

    prompt = f"""
    You are an expert document assistant.
    Answer the user's question accurately using ONLY the information provided in the Document Context below.
    If the exact details are not found in the context, state clearly what is missing rather than giving up completely.

    Document Context:
    {combined_context}

    Question:
    {user_query}
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )

    return response.text


# ============================================================
# 8. USER INTERFACE
# ============================================================

st.title("📄 Native Gemini - PDF RAG Assistant")
st.write("Upload a PDF, store its embeddings in Neon PostgreSQL, and ask questions about it.")

with st.sidebar:
    st.header("⚙️ System Monitor")
    st.success("✅ Native Gemini API Active")
    st.info("Embedding Model: gemini-embedding-001")
    st.info("Embedding dimensions: 768")
    st.info("LLM Model: gemini-3.5-flash")
    st.success("✅ Neon PostgreSQL Connected")

uploaded_file = st.file_uploader("📂 Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.write(f"Selected file: **{uploaded_file.name}**")
    if st.button("🚀 Process & Index Document"):
        with st.spinner("Extracting PDF and generating Gemini embeddings..."):
            try:
                num_chunks = ingest_pdf_to_db(uploaded_file)
                st.success("✅ Document processed successfully!")
                st.info(f"Created and stored **{num_chunks} chunks**.")
            except Exception as e:
                st.error("❌ Document ingestion failed.")
                st.exception(e)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about your uploaded document...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            try:
                answer = ask_rag(user_input)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("❌ Failed to generate answer.")
                st.exception(e)
