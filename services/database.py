import psycopg2

def get_db_connection(db_url: str):
    return psycopg2.connect(db_url)

def initialize_database(db_url: str):
    """Ensure vector extension and target schema exist."""
    conn = None
    try:
        conn = get_db_connection(db_url)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdf_documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding VECTOR(768)
                );
            """)
        conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def search_similar_chunks(db_url: str, embedding_string: str, top_k: int = 5) -> list[str]:
    """Perform cosine vector similarity search using pgvector (<=>)."""
    conn = None
    try:
        conn = get_db_connection(db_url)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT content FROM pdf_documents
                ORDER BY embedding <=> %s::vector LIMIT %s;
            """, (embedding_string, top_k))
            results = cur.fetchall()
            return [row[0] for row in results]
    finally:
        if conn:
            conn.close()

def insert_document_chunks(db_url: str, chunks_with_embeddings: list[tuple[str, str]]):
    """Recreate target table and batch insert new document vectors."""
    conn = None
    try:
        conn = get_db_connection(db_url)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("DROP TABLE IF EXISTS pdf_documents;")
            cur.execute("""
                CREATE TABLE pdf_documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding VECTOR(768)
                );
            """)
            for content, embedding_str in chunks_with_embeddings:
                cur.execute("""
                    INSERT INTO pdf_documents (content, embedding)
                    VALUES (%s, %s::vector);
                """, (content, embedding_str))
        conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
