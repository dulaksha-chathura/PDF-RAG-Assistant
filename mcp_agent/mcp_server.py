import os
import psycopg2
from fastmcp import FastMCP
from google import genai

# Initialize FastMCP Server
mcp = FastMCP("PDF-RAG-MCP-Server")

# Initialize Gemini Client and Database Connection String
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
NEON_URL = os.getenv("NEON_DATABASE_URL")

@mcp.tool()
def search_pdf_documents(query: str, top_k: int = 5) -> str:
    """Search indexed PDF document chunks in Neon PostgreSQL using pgvector similarity search."""
    if not NEON_URL:
        return "Error: NEON_DATABASE_URL environment variable is missing."

    # 1. Generate Query Embedding via Gemini API
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )
    query_embedding = response.embedding.values

    # 2. Query Neon PostgreSQL (pgvector)
    conn = psycopg2.connect(NEON_URL)
    cursor = conn.cursor()
    
    query_sql = """
        SELECT content, 1 - (embedding <=> %s::vector) AS similarity
        FROM pdf_documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    cursor.execute(query_sql, (query_embedding, query_embedding, top_k))
    rows = cursor.fetchall()
    
    cursor.close()
    conn.close()

    if not rows:
        return "No relevant context found in the PDF documents."

    # 3. Format results for the Agent
    results = [f"[Similarity: {r[1]:.2f}]\n{r[0]}" for r in rows]
    return "\n---\n".join(results)

if __name__ == "__main__":
    mcp.run()
