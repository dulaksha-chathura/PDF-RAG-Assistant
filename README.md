# 📄 PDF RAG Assistant (Native & Agentic Architecture)

A lightweight, high-performance **PDF RAG (Retrieval-Augmented Generation)** assistant built with **Streamlit**, **Google Gemini**, and **Neon PostgreSQL (`pgvector`)**.

The repository supports both a **Native Execution Mode** for simple deployment and an **Agentic Microservices Mode** utilizing **FastMCP** and **LangChain** for decoupled, tool-driven document reasoning.

---

## ✨ Features

* 📄 **Local PDF Ingestion:** Fast multi-page PDF text extraction using `pypdf`.
* 🔍 **Vector Similarity Search:** High-accuracy semantic retrieval via `pgvector` and 768-dimensional `gemini-embedding-001` embeddings.
* 🤖 **Dual Architecture:**

  * **Native Mode:** Lightweight, direct integration with Gemini.
  * **Agentic Mode:** Decoupled FastMCP tool server orchestrated via a LangChain agent reasoning loop.
* 🔒 **Data Privacy:** Local document parsing ensures raw PDF contents remain secure on your machine.

---

## 🏗️ Architecture Modes

### 1. Native Execution Pipeline

```text
[Document Ingestion]
PDF → pypdf → Text Chunks → Gemini Embeddings → Neon PostgreSQL (pgvector)

[Question Answering]
User Question → Gemini Embedding → pgvector Search → Context Chunks → Gemini Flash → Answer
```

### 2. Agentic Microservice Pipeline

```text
[Decoupled Tool Layer]
FastMCP Server → search_pdf_documents Tool → Neon PostgreSQL (pgvector)

[Agentic Reasoning Loop]
Streamlit UI → LangChain Agent → JSON-RPC → FastMCP Server → Grounded Answer
```

---

## 🧰 Tech Stack

* **Language & Framework:** Python 3.10+, Streamlit
* **Agentic Framework:** LangChain (`langchain-google-genai`, `langchain-mcp-adapters`)
* **Tool Server:** FastMCP (`fastmcp`, `mcp`)
* **LLM & Embeddings:** Google Gemini API (`gemini-1.5-flash`, `gemini-embedding-001`)
* **Vector Database:** Neon PostgreSQL with `pgvector`
* **PDF Parsing:** `pypdf`

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/dulaksha-chathura/PDF-RAG-Assistant.git
cd PDF-RAG-Assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Credentials

Create `.streamlit/secrets.toml` or a local `.env` file:

```env
GEMINI_API_KEY = "your-gemini-api-key"
NEON_DATABASE_URL = "your-neon-postgresql-url"
```

---

## 🗄️ Database Setup

The application automatically enables `pgvector` and initializes the table upon launch:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS pdf_documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(768)
);
```

---

## 🏃 Execution Options

### Option A: Run Native Streamlit App

```bash
streamlit run app.py
```

### Option B: Run Agentic Mode

**1. Start the FastMCP Server (Terminal 1):**

```bash
python mcp_server.py
```

**2. Run the Agentic Streamlit UI (Terminal 2):**

```bash
streamlit run app.py -- --mode agentic
```



