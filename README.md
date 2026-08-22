# 📄 Native Gemini PDF RAG Assistant

A lightweight, high-performance **Retrieval-Augmented Generation (RAG)** system built with **Streamlit**, the **Google Gemini API** (`gemini-3.5-flash` and `gemini-embedding-001`), and **Neon PostgreSQL** with `pgvector`.

Upload multi-page PDFs, generate and store vector embeddings in Neon, and ask grounded questions directly against your document context.

---

## 🌟 Key Features

* **Grounded Document Q&A:** Uses `gemini-3.5-flash` to answer questions based strictly on the uploaded document context, reducing hallucinations.
* **Vector Similarity Search:** Powered by Neon PostgreSQL with the `pgvector` extension using **Cosine Distance** (`<=>`).
* **Optimized Embedding Pipeline:** Uses `gemini-embedding-001` with an explicit **768-dimensional vector output**.
* **Local PDF Parsing:** PDF text is extracted locally using `pypdf`; the physical PDF file itself is never uploaded to the Gemini API.
* **Low Mobile Data Footprint:** Approximately **15–23 MB** of network data to index a 1,000-page document and approximately **0.03 MB per query**, depending on chunking and API request sizes.

---

## 🏗️ System Architecture

### 📥 Ingestion Pipeline

```text
PDF File
   │
   ▼
┌──────────┐
│  pypdf   │
└────┬─────┘
     │
     ▼
Text Chunks
     │
     ▼
┌─────────────────────────┐
│ gemini-embedding-001    │
│ output_dimensionality   │
│ = 768                   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Neon PostgreSQL         │
│ + pgvector              │
└─────────────────────────┘
```

### 🔎 Query Pipeline

```text
User Question
      │
      ▼
┌─────────────────────────┐
│ gemini-embedding-001    │
└───────────┬─────────────┘
            │
            ▼
   Vector Similarity Search
            │
            ▼
       Top 5 Chunks
            │
            ▼
     Context Prompt
            │
            ▼
┌─────────────────────────┐
│ gemini-3.5-flash        │
└───────────┬─────────────┘
            │
            ▼
          Answer
```

---

## 🧰 Technical Stack

| Component           | Technology             |
| ------------------- | ---------------------- |
| Frontend / UI       | Streamlit              |
| Application         | Python                 |
| LLM                 | `gemini-3.5-flash`     |
| Embedding Model     | `gemini-embedding-001` |
| Embedding Dimension | 768                    |
| PDF Parser          | `pypdf`                |
| Vector Database     | Neon PostgreSQL        |
| Vector Extension    | `pgvector`             |
| Database Driver     | `psycopg2-binary`      |
| Gemini SDK          | `google-genai`         |

---

## 🚀 Quickstart

### 1. Prerequisites

Make sure you have:

* Python **3.10 or higher**
* A **Google Gemini API key**
* A **Neon PostgreSQL** database
* `pgvector` support enabled in the Neon database

---

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/gemini-pdf-rag.git
cd gemini-pdf-rag
```

---

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## 📦 Dependencies

Create a `requirements.txt` file containing:

```text
streamlit
psycopg2-binary
pypdf
google-genai
```

Alternatively:

```bash
pip install streamlit psycopg2-binary pypdf google-genai
```

---

## 🔐 Configuration

Create the following file:

```text
.streamlit/
└── secrets.toml
```

Add your database connection string and Gemini API key:

```toml
NEON_DATABASE_URL = "postgresql://user:password@ep-your-endpoint.neon.tech/neondb?sslmode=require"
GEMINI_API_KEY = "your-gemini-api-key-here"
```

### ⚠️ Security

Never commit `secrets.toml` to GitHub.

Add it to `.gitignore`:

```gitignore
.streamlit/secrets.toml
.env
__pycache__/
*.pyc
```

---

## ▶️ Run the Application

Launch the Streamlit application with:

```bash
streamlit run app.py
```

After starting, Streamlit will provide a local URL such as:

```text
http://localhost:8501
```

Open the URL in your browser to use the RAG assistant.

---

# 🛠️ Database Initialization

The application automatically checks for the `pgvector` extension and creates the required table when it starts.

### Enable `pgvector`

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Create the Documents Table

```sql
CREATE TABLE IF NOT EXISTS pdf_documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(768)
);
```

The `embedding` column stores the **768-dimensional Gemini embeddings**.

---

## 🔍 Vector Similarity Search

The system uses PostgreSQL's `pgvector` extension to perform vector similarity searches.

For example:

```sql
SELECT
    id,
    content,
    embedding <=> %s::vector AS distance
FROM pdf_documents
ORDER BY embedding <=> %s::vector
LIMIT 5;
```

The `<=>` operator calculates **Cosine Distance** between the stored document embedding and the query embedding.

The five most relevant chunks are then passed to the Gemini language model as context.

---

# 🧠 RAG Workflow

The complete workflow can be summarized as follows:

```text
                 DOCUMENT INGESTION
                         │
                         ▼
                    Upload PDF
                         │
                         ▼
                  Extract PDF Text
                       (pypdf)
                         │
                         ▼
                    Split into
                    Text Chunks
                         │
                         ▼
                  Generate Embeddings
                 (gemini-embedding-001)
                         │
                         ▼
                  Store in Neon DB
                    (pgvector)


                    USER QUERY
                         │
                         ▼
                   User Question
                         │
                         ▼
                  Generate Query
                    Embedding
                         │
                         ▼
               Vector Similarity Search
                         │
                         ▼
                  Retrieve Top 5
                   Relevant Chunks
                         │
                         ▼
                   Build Context
                       Prompt
                         │
                         ▼
                  gemini-3.5-flash
                         │
                         ▼
                    Final Answer
```

---

## 📊 Bandwidth & Network Data Profile

The PDF itself is parsed locally, so network traffic primarily consists of text chunks sent for embedding, embedding responses, database operations, and query/response communication.

| Action                                         | Estimated Mobile Data |
| ---------------------------------------------- | --------------------: |
| Parsing 1 PDF page                             |             ~20–50 KB |
| Indexing a 1,000-page document (~2,500 chunks) |             ~15–23 MB |
| Asking 1 question                              |              ~0.03 MB |

> **Note:** Actual bandwidth usage depends on PDF text density, chunk size, number of chunks, API request batching, database communication, and generated response length.

---

# 📁 Project Structure

A typical project structure is:

```text
gemini-pdf-rag/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── secrets.toml
```

---

## 💡 Why RAG?

A traditional LLM answers questions using knowledge learned during model training. This can lead to incorrect answers when the information is specific to a private document.

RAG adds a retrieval step:

```text
Document
   ↓
Embedding
   ↓
Vector Database
   ↓
Relevant Context
   ↓
LLM
   ↓
Grounded Answer
```

Instead of asking the LLM to answer from its general knowledge, the application retrieves relevant sections from the uploaded document and provides them as context.

---

## 🔒 Privacy Considerations

The application extracts PDF text locally using `pypdf`.

The original PDF file is therefore **not directly uploaded to Gemini**. However, extracted text is sent to the Gemini API during the embedding process, and user questions/context are sent during question answering.

For sensitive documents, review the applicable **Google Gemini API and Neon PostgreSQL data-handling policies** before deployment.

---

## ⚡ Performance Characteristics

The system is designed to keep the RAG pipeline lightweight:

* Local PDF processing
* Compact text chunks
* 768-dimensional embeddings
* PostgreSQL-based vector search
* Top-5 context retrieval
* Gemini-based answer generation
* Minimal client-side data transfer

---

## 🔧 Possible Future Improvements

The system can be extended with:

* [ ] PDF page-number tracking
* [ ] Source citations in generated answers
* [ ] Metadata filtering
* [ ] Multiple-document support
* [ ] Document deletion
* [ ] Duplicate document detection
* [ ] Improved chunking strategies
* [ ] Batch embedding requests
* [ ] Conversation history
* [ ] Streaming Gemini responses
* [ ] Authentication and user accounts
* [ ] Hybrid keyword + vector search
* [ ] Reranking of retrieved chunks
* [ ] OCR support for scanned PDFs
* [ ] Document management dashboard

---

## 🧪 Example Usage

### 1. Upload a PDF

Select a PDF document through the Streamlit interface.

### 2. Index the Document

The application:

1. Extracts text locally using `pypdf`.
2. Splits the text into chunks.
3. Generates embeddings using `gemini-embedding-001`.
4. Stores the chunks and embeddings in Neon PostgreSQL.

### 3. Ask a Question

For example:

```text
What is the main objective of this document?
```

### 4. Retrieve Relevant Context

The query is converted into an embedding and compared against the stored document embeddings.

The most relevant five chunks are retrieved.

### 5. Generate the Answer

The retrieved context is passed to `gemini-3.5-flash`, which generates the final answer based on the available document context.

---

## 📜 License

This project is licensed under the **MIT License**.

```text
MIT License

Copyright (c) 2026 Dulaksha Chathura

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## ⭐ Acknowledgements

Built using:

* **Streamlit** — Web application framework
* **Google Gemini API** — Embeddings and language generation
* **Neon PostgreSQL** — Serverless PostgreSQL database
* **pgvector** — Vector similarity search
* **pypdf** — Local PDF text extraction

---

## 📌 Summary

**Native Gemini PDF RAG Assistant** combines local PDF processing, Gemini embeddings, PostgreSQL vector search, and Gemini-powered generation into a lightweight document-question-answering system.

```text
PDF
 │
 ▼
pypdf
 │
 ▼
Text Chunks
 │
 ▼
Gemini Embeddings
 │
 ▼
Neon + pgvector
 │
 ▼
Similarity Search
 │
 ▼
Relevant Context
 │
 ▼
Gemini Flash
 │
 ▼
Grounded Answer
```

The result is a simple and efficient RAG architecture that can be deployed with **Python, Streamlit, Gemini, and Neon PostgreSQL**.

