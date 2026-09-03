import os
import sys

import psycopg2
from pypdf import PdfReader
from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP


# ============================================================================
# MCP SERVER
# ============================================================================

mcp = FastMCP("PDF Knowledge Base Server")


# ============================================================================
# CONFIGURATION
# ============================================================================

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768

CHUNK_SIZE = 500

TOP_K = 4


# ============================================================================
# DEBUG LOGGING
# ============================================================================

def log_debug(msg: str):
    """
    Safely logs debug messages to stderr.

    IMPORTANT:
    MCP uses stdout for JSON-RPC communication.
    Therefore, debug messages must NEVER be printed to stdout.
    """

    sys.stderr.write(
        f"[MCP SERVER DEBUG] {msg}\n"
    )

    sys.stderr.flush()


# ============================================================================
# GEMINI CLIENT
# ============================================================================

def get_gemini_client():

    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:

        raise ValueError(
            "Missing GEMINI_API_KEY environment variable."
        )

    return genai.Client(
        api_key=gemini_api_key
    )


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def get_db_connection():

    neon_db_url = os.getenv(
        "NEON_DATABASE_URL"
    )

    if not neon_db_url:

        raise ValueError(
            "Missing NEON_DATABASE_URL environment variable."
        )

    return psycopg2.connect(
        neon_db_url
    )


# ============================================================================
# GENERATE GEMINI EMBEDDING
# ============================================================================

def generate_embedding(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[float]:
    """
    Generate a 768-dimensional embedding using
    Gemini gemini-embedding-001.

    The same embedding model and dimensionality
    must be used for both documents and queries.
    """

    if not text or not text.strip():

        raise ValueError(
            "Cannot generate embedding for empty text."
        )

    client = get_gemini_client()

    try:

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,

            contents=text,

            config=types.EmbedContentConfig(

                output_dimensionality=EMBEDDING_DIMENSION,

                task_type=task_type
            )
        )

        if not result.embeddings:

            raise RuntimeError(
                "Gemini returned no embeddings."
            )

        embedding = result.embeddings[0].values

        if embedding is None:

            raise RuntimeError(
                "Gemini returned an empty embedding."
            )

        embedding = list(embedding)

        # ------------------------------------------------------------
        # Verify vector dimension
        # ------------------------------------------------------------

        if len(embedding) != EMBEDDING_DIMENSION:

            raise RuntimeError(
                f"Invalid embedding dimension. "
                f"Expected {EMBEDDING_DIMENSION}, "
                f"received {len(embedding)}."
            )

        return embedding

    except Exception as e:

        log_debug(
            f"Gemini embedding error: {str(e)}"
        )

        raise


# ============================================================================
# CONVERT PYTHON VECTOR → PGVECTOR STRING
# ============================================================================

def vector_to_pgvector(
    vector: list[float]
) -> str:

    return (
        "["
        + ",".join(
            str(float(value))
            for value in vector
        )
        + "]"
    )


# ============================================================================
# PDF INGESTION
# ============================================================================

def ingest_pdf_to_database(
    file_path: str,
    original_filename: str = None
) -> bool:
    """
    Parse PDF, create chunks, generate Gemini embeddings,
    and store the vectors in Neon PostgreSQL.
    """

    gemini_api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not gemini_api_key:

        log_debug(
            "Ingestion aborted: "
            "Missing GEMINI_API_KEY."
        )

        return False

    if not os.path.exists(file_path):

        log_debug(
            f"Ingestion aborted: "
            f"File does not exist: {file_path}"
        )

        return False

    file_name = (
        original_filename
        if original_filename
        else os.path.basename(file_path)
    )

    conn = None
    cursor = None

    try:

        # ============================================================
        # STEP 1 — PDF PARSING
        # ============================================================

        log_debug(
            f"Parsing PDF: {file_name}"
        )

        reader = PdfReader(
            file_path
        )

        full_text = ""

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text()

            if text:

                full_text += (
                    text
                    + "\n"
                )

        if not full_text.strip():

            log_debug(
                "No text could be extracted from PDF."
            )

            return False

        # ============================================================
        # STEP 2 — TEXT CHUNKING
        # ============================================================

        chunks = [
            full_text[i:i + CHUNK_SIZE]
            for i in range(
                0,
                len(full_text),
                CHUNK_SIZE
            )
        ]

        log_debug(
            f"Generated {len(chunks)} text chunks."
        )

        # ============================================================
        # STEP 3 — DATABASE CONNECTION
        # ============================================================

        conn = get_db_connection()

        cursor = conn.cursor()

        # ============================================================
        # STEP 4 — ENABLE PGVECTOR
        # ============================================================

        cursor.execute(
            """
            CREATE EXTENSION IF NOT EXISTS vector;
            """
        )

        # ============================================================
        # STEP 5 — CREATE TABLE
        # ============================================================

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS pdf_chunks (
                id SERIAL PRIMARY KEY,
                file_name TEXT,
                content TEXT,
                embedding VECTOR({EMBEDDING_DIMENSION})
            );
            """
        )

        # ============================================================
        # STEP 6 — GENERATE EMBEDDINGS + INSERT
        # ============================================================

        inserted_count = 0

        for index, chunk in enumerate(
            chunks
        ):

            if not chunk.strip():

                continue

            log_debug(
                f"Embedding chunk "
                f"{index + 1}/{len(chunks)}..."
            )

            vector = generate_embedding(
                chunk,
                task_type="RETRIEVAL_DOCUMENT"
            )

            vector_str = vector_to_pgvector(
                vector
            )

            cursor.execute(
                """
                INSERT INTO pdf_chunks
                (
                    file_name,
                    content,
                    embedding
                )
                VALUES
                (
                    %s,
                    %s,
                    %s::vector
                );
                """,
                (
                    file_name,
                    chunk,
                    vector_str
                )
            )

            inserted_count += 1

        # ============================================================
        # STEP 7 — COMMIT
        # ============================================================

        conn.commit()

        log_debug(
            f"Successfully inserted "
            f"{inserted_count} vector rows "
            f"for {file_name}."
        )

        return True

    except Exception as e:

        if conn:

            conn.rollback()

        log_debug(
            f"Error during ingestion: {str(e)}"
        )

        return False

    finally:

        if cursor:

            cursor.close()

        if conn:

            conn.close()


# ============================================================================
# MCP TOOL — VECTOR SEARCH
# ============================================================================

@mcp.tool()
def search_pdf_knowledge_base(
    query: str
) -> str:
    """
    Searches pre-indexed PDF chunks stored in
    Neon PostgreSQL using cosine similarity.
    """

    if not query or not query.strip():

        return (
            "Error: Search query cannot be empty."
        )

    conn = None
    cursor = None

    try:

        log_debug(
            f"Querying knowledge base: '{query}'"
        )

        # ============================================================
        # STEP 1 — EMBED USER QUERY
        # ============================================================

        query_vector = generate_embedding(
            query,
            task_type="RETRIEVAL_QUERY"
        )

        vector_str = vector_to_pgvector(
            query_vector
        )

        # ============================================================
        # STEP 2 — DATABASE CONNECTION
        # ============================================================

        conn = get_db_connection()

        cursor = conn.cursor()

        # ============================================================
        # STEP 3 — COSINE SIMILARITY SEARCH
        # ============================================================

        cursor.execute(
            """
            SELECT
                file_name,
                content,
                1 - (embedding <=> %s::vector)
                    AS similarity

            FROM pdf_chunks

            ORDER BY embedding <=> %s::vector

            LIMIT %s;
            """,
            (
                vector_str,
                vector_str,
                TOP_K
            )
        )

        rows = cursor.fetchall()

        # ============================================================
        # STEP 4 — NO RESULTS
        # ============================================================

        if not rows:

            return (
                "No relevant information found "
                "in the knowledge base."
            )

        # ============================================================
        # STEP 5 — FORMAT RESULTS FOR AGENT
        # ============================================================

        results = []

        for file_name, content, similarity in rows:

            if not content:

                continue

            results.append(
                f"[Source: {file_name}]\n"
                f"[Similarity: {similarity:.4f}]\n"
                f"{content}"
            )

        if not results:

            return (
                "No relevant information found "
                "in the knowledge base."
            )

        return "\n\n---\n\n".join(
            results
        )

    except Exception as e:

        log_debug(
            f"Search error: {str(e)}"
        )

        return (
            f"Database query error: {str(e)}"
        )

    finally:

        if cursor:

            cursor.close()

        if conn:

            conn.close()


# ============================================================================
# MCP SERVER ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    log_debug(
        "Starting PDF Knowledge Base MCP Server..."
    )

    log_debug(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    log_debug(
        f"Embedding dimensions: "
        f"{EMBEDDING_DIMENSION}"
    )

    mcp.run(
        transport="stdio"
    )
