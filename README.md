# 📄 PDF RAG Assistant (Native 3-Tier Architecture)

A lightweight, high-performance **PDF RAG (Retrieval-Augmented Generation)** assistant built with **Streamlit**, **Google Gemini**, and **Neon PostgreSQL (`pgvector`)**.

The application features a clean **3-Tier Modular Architecture** separating the presentation interface, core business logic/AI services, and persistent vector storage.

---

<p align="center">
  <img src="assets/native_app.jpeg" alt="Native Streamlit App" width="100%">
  <br>
  <em>Figure: Native Streamlit App Interface</em>
</p>

---

## ✨ Features

* 📄 **Local PDF Processing:** Fast multi-page PDF text extraction and chunking using `pypdf`.
* 🔍 **Vector Similarity Search:** High-accuracy semantic retrieval via `pgvector` and 768-dimensional `gemini-embedding-001` embeddings.
* ⚡ **Fast Generation:** Grounded answer generation using `gemini-3.5-flash`.
* 🏗️ **3-Tier Modular Structure:** Separated code layers (UI, Services, and Database) for maintainability, testing, and scalability.
* 🔒 **Data Privacy:** Local document parsing ensures raw PDF contents remain secure on your machine before vector indexing.

---

## 🏗️ Architecture & Data Flow

```text
[ TIER 1: PRESENTATION LAYER ]
          Streamlit UI (User Input & Chat Display)
                            │
                            ▼
[ TIER 2: APPLICATION SERVICES LAYER ]
   ├─► PDF Processor (pypdf text chunking)
   ├─► Embedding Service (Google Gemini gemini-embedding-001)
   └─► RAG Orchestrator (Google Gemini gemini-3.5-flash)
                            │
                            ▼
[ TIER 3: DATA ACCESS LAYER ]
   └─► Neon PostgreSQL Cloud Database (pgvector Cosine Distance Search)
```

### End-to-End Pipeline

1. **Ingestion:** Uploaded PDFs are parsed and split into overlapping chunks, vectorized via `gemini-embedding-001`, and stored in Neon PostgreSQL.
2. **Retrieval:** User queries are converted to query vectors and matched against document vectors using `pgvector` cosine distance (`<=>`).
3. **Generation:** Retrieved context chunks and the user query are formatted into a grounded prompt and processed by `gemini-3.5-flash`.

---

## 🧰 Tech Stack

* **UI & Presentation:** Streamlit 1.30+
* **Language & Runtime:** Python 3.10+
* **LLM & Embeddings:** Google Gemini API (`gemini-3.5-flash`, `gemini-embedding-001`)
* **Vector Database:** Neon PostgreSQL with `pgvector` extension
* **Database Driver:** `psycopg2-binary`
* **PDF Parsing:** `pypdf`

---

## 📁 Repository Structure

```text
PDF-RAG-Assistant/
├── assets/
│   └── native_app.jpeg            # Application interface screenshot
├── services/
│   ├── __init__.py                # Package marker
│   ├── database.py                # PostgreSQL connection & pgvector queries
│   ├── embeddings.py              # Gemini embedding API integration
│   └── pdf_processor.py           # PyPDF text extraction & chunking
├── ui/
│   ├── __init__.py                # Package marker
│   ├── sidebar.py                 # System monitor sidebar
│   └── chat.py                    # Streamlit chat state and renderer
├── .gitignore                     # Shields secrets & bytecode
├── app.py                         # Application entry point & orchestrator
├── README.md                      # Project documentation
└── requirements.txt               # Dependencies
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository

```bash
git clone https://github.com/dulaksha-chathura/PDF-RAG-Assistant.git
cd PDF-RAG-Assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Local Credentials

Create a directory named `.streamlit` and add a `secrets.toml` file inside it:

```bash
mkdir .streamlit
```

Inside `.streamlit/secrets.toml`, add your credentials:

```toml
NEON_DATABASE_URL = "postgresql://user:password@ep-cool-db-123456.us-east-1.aws.neon.tech/neondb?sslmode=require"
GEMINI_API_KEY = "your-gemini-api-key-here"
```

> ⚠️ **Security Note:** Never commit `.streamlit/secrets.toml` to GitHub. Keep it listed in `.gitignore`.

---

## 🗄️ Database Initialization

The application automatically checks for necessary extensions and schema setup upon launch. If you prefer manual database setup, run the following SQL statements in your Neon SQL Editor:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create PDF documents table
CREATE TABLE IF NOT EXISTS pdf_documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(768)
);
```

---

## 🏃 Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

Access the app in your browser at `http://localhost:8501`.

1. Upload a PDF file in the sidebar/uploader.
2. Click **Process & Index Document** to generate embeddings and store them in Neon DB.
3. Type your question in the chat bar to receive grounded answers!




