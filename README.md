```markdown
# 📄 Native Gemini PDF RAG Assistant

A lightweight, high-performance Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **Google Gemini API** (`gemini-3.5-flash` & `gemini-embedding-001`), and **Neon PostgreSQL** (`pgvector`). Upload multi-page PDFs, generate and store vector embeddings in Neon, and ask grounded questions directly against your document context.

---

## 🌟 Key Features

* **Grounded Document Q&A:** Uses `gemini-3.5-flash` to answer questions strictly based on uploaded document context, preventing hallucinations.
* **Vector Similarity Search:** Powered by Neon PostgreSQL with the `pgvector` extension using Cosine Distance (`<=>`).
* **Optimized Embedding Pipeline:** Utilizes `gemini-embedding-001` configured with explicit 768-dimensional vector output.
* **Local Parsing:** Extracted via `pypdf` locally on your machine—the physical PDF file is never sent across the network.
* **Low Mobile Data Footprint:** Extremely bandwidth-efficient (~15–23 MB to index a 1,000-page document; ~0.03 MB per query).

---

## 🏗️ System Architecture


```

```
                   [ INGESTION PIPELINE ]

```

PDF File ──> (pypdf) ──> Text Chunks ──> (gemini-embedding-001) ──> Neon DB (pgvector)

```
                      [ QUERY PIPELINE ]

```

Question ──> (gemini-embedding-001) ──> Top 5 Search ──> Context Prompt ──> (gemini-3.5-flash) ──> Answer

```

### Technical Stack
* **Frontend UI:** Streamlit (`app.py`)
* **LLM Engine:** `gemini-3.5-flash`
* **Embedding Model:** `gemini-embedding-001` (`output_dimensionality=768`)
* **Vector Storage:** Neon PostgreSQL with `pgvector`

---

## 🚀 Quickstart

### 1. Prerequisites
* Python 3.10 or higher
* A Google Gemini API Key
* A serverless Neon PostgreSQL database instance with `pgvector` support

### 2. Installation
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/gemini-pdf-rag.git](https://github.com/your-username/gemini-pdf-rag.git)
cd gemini-pdf-rag
pip install -r requirements.txt

```

### 3. Dependencies (`requirements.txt`)

Ensure your `requirements.txt` includes:

```text
streamlit
psycopg2-binary
pypdf
google-genai

```

### 4. Configuration

Create a `.streamlit/secrets.toml` file in your root project folder:

```toml
NEON_DATABASE_URL = "postgresql://user:password@ep-your-endpoint.neon.tech/neondb?sslmode=require"
GEMINI_API_KEY = "your-gemini-api-key-here"

```

### 5. Run Application

Launch the Streamlit web app:

```bash
streamlit run app.py

```

---

## 🛠️ Database Initialization

The application automatically executes vector extension checks and table creation upon startup:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS pdf_documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(768)
);

```

---

## 📊 Bandwidth & Network Data Profile

| Action | Estimated Mobile Data |
| --- | --- |
| **Parsing 1 PDF Page** | ~20 KB – 50 KB |
| **Indexing a 1,000-Page Document** (~2,500 chunks) | ~15 MB – 23 MB |
| **Asking 1 Question** | ~0.03 MB |

---

## 📜 License

This project is licensed under the MIT License.

```

```
