from google.genai import types

def get_embedding(gemini_client, text_chunk: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Generate 768-dimensional embeddings using Gemini embedding API."""
    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text_chunk,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768
        )
    )
    return response.embeddings[0].values

def vector_to_pgvector(vector: list[float]) -> str:
    """Format float array into Postgres pgvector string syntax: [x,y,z]."""
    return "[" + ",".join(map(str, vector)) + "]"
