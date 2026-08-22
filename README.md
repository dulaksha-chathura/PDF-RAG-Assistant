# 📄 Native Gemini PDF RAG Assistant

A lightweight **PDF RAG (Retrieval-Augmented Generation)** assistant built with **Streamlit**, **Google Gemini**, and **Neon PostgreSQL + pgvector**.

Upload a PDF and ask questions about its contents using semantic vector search and Gemini.

## ✨ Features

* 📄 Upload and process multi-page PDFs
* 🔍 Semantic search using `pgvector`
* 🤖 Gemini-powered question answering
* 🧠 `gemini-embedding-001` embeddings with 768 dimensions
* 🔒 PDF text extraction performed locally using `pypdf`
* 📱 Designed for low network usage

## 🏗️ Architecture

### Document Ingestion

```text
PDF → pypdf → Text Chunks → Gemini Embeddings → Neon PostgreSQL
```

### Question Answering

```text
Question → Gemini Embedding → Vector Search → Top 5 Chunks → Gemini Flash → Answer
```

## 🧰 Requirements

* Python 3.10+
* Google Gemini API key
* Neon PostgreSQL database with `pgvector`

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/gemini-pdf-rag.git
cd gemini-pdf-rag
```

### 2. Install Dependencies

```bash
pip install streamlit psycopg2-binary pypdf google-genai
```

### 3. Configure Credentials

Create:

```text
.streamlit/secrets.toml
```

Add your credentials:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
NEON_DATABASE_URL = "your-neon-postgresql-url"
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open the Streamlit URL shown in the terminal and start using the PDF RAG assistant.

## 🗄️ Database

The application automatically enables `pgvector` and creates the required table when it starts:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS pdf_documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(768)
);
```

## 🧰 Tech Stack

* **Python**
* **Streamlit**
* **Google Gemini API**

  * `gemini-3.5-flash`
  * `gemini-embedding-001`
* **Neon PostgreSQL**
* **pgvector**
* **pypdf**


