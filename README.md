# 📄 PDF RAG Assistant (Native & Agentic Architecture)

A lightweight, high-performance **PDF RAG (Retrieval-Augmented Generation)** assistant built with **Streamlit**, **Google Gemini**, and **Neon PostgreSQL (`pgvector`)**.

The repository supports both a **Native Execution Mode** for simple direct deployment and an **Agentic Microservices Mode** utilizing **FastMCP** and **LangChain** for decoupled, tool-driven document reasoning.

---

<p align="center">
  <img src="assets/native_app.jpeg" alt="Native Streamlit App" width="100%">
  <br>
  <em>Figure: Streamlit Application Interface</em>
</p>

---

## ✨ Features

* 📄 **Local PDF Ingestion:** Fast multi-page PDF text extraction and chunking using `pypdf`.
* 🔍 **Vector Similarity Search:** High-accuracy semantic retrieval via `pgvector` and 768-dimensional `gemini-embedding-001` embeddings.
* 🤖 **Dual Execution Architecture:**

  * **Native Mode:** Direct integration with Gemini API and local vector retrieval.
  * **Agentic Mode:** Decoupled FastMCP tool server orchestrated via a LangChain ReAct agent loop.
* 🔒 **Data Privacy:** Local document processing ensures raw PDF contents are indexed safely to your Neon vector database.

---

## 🏗️ Architecture Modes

### 1. Native Execution Pipeline

```text
[Document Ingestion]
PDF → pypdf → Text Chunks → Gemini Embeddings → Neon PostgreSQL (pgvector)

[Question Answering]
User Question → Gemini Embedding → pgvector Search → Context Chunks → Gemini Flash → Answer
```

### 2. Agentic Microservice Pipeline (MCP)

```text
[Decoupled Tool Layer]
FastMCP Server → search_pdf_documents Tool → Neon PostgreSQL (pgvector)

[Agentic Reasoning Loop]
Streamlit UI (app_mcp.py) → LangChain ReAct Agent → Stdio JSON-RPC → FastMCP Server → Grounded Answer
```

---

## 🧰 Tech Stack

* **UI & Presentation:** Streamlit 1.30+
* **Language & Runtime:** Python 3.10+
* **Agentic Framework:** LangChain (`langchain-google-genai`, `langchain-mcp-adapters`, `langgraph`)
* **Tool Server:** FastMCP (`fastmcp`, `mcp`)
* **LLM & Embeddings:** Google Gemini API (`gemini-3.5-flash`, `gemini-embedding-001`)
* **Vector Database:** Neon PostgreSQL with `pgvector`
* **PDF Parsing:** `pypdf`

---

## 📁 Repository Structure

```text
PDF-RAG-Assistant/
├── assets/
│   └── native_app.jpeg            # Application screenshot
├── mcp_agent/
│   ├── __init__.py                # Package marker for MCP module
│   ├── agent.py                   # LangChain ReAct agent orchestrator
│   └── mcp_server.py              # FastMCP server exposing pgvector tool
├── services/
│   ├── __init__.py                # Package marker for Services module
│   ├── database.py                # PostgreSQL connection & pgvector queries
│   ├── embeddings.py              # Gemini embedding generation
│   └── pdf_processor.py           # PyPDF text extraction & chunking
├── ui/
│   ├── __init__.py                # Package marker for UI module
│   ├── sidebar.py                 # System monitor sidebar UI
│   └── chat.py                    # Streamlit chat renderer & state manager
├── .gitignore                     # Git ignore rules
├── app.py                         # Native Streamlit entry point
├── app_mcp.py                     # Agentic MCP Streamlit entry point
├── README.md                      # Documentation
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

### 3. Configure Credentials

Create a directory named `.streamlit` and add a `secrets.toml` file:

```bash
mkdir .streamlit
```

Add your database and API credentials inside `.streamlit/secrets.toml`:

```toml
NEON_DATABASE_URL = "postgresql://user:password@ep-cool-db-123456.us-east-1.aws.neon.tech/neondb?sslmode=require"
GEMINI_API_KEY = "your-gemini-api-key-here"
```

---

## 🗄️ Database Initialization

The application automatically checks and initializes the database on startup. For manual setup in Neon's SQL Editor:

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

Choose which mode you want to launch:

### Option A: Native Mode (Direct Pipeline)

Runs the direct Streamlit app using Gemini and PostgreSQL vector search:

```bash
streamlit run app.py
```

### Option B: Agentic MCP Mode (FastMCP + LangChain)

Runs the agentic version where the Streamlit UI delegates tool execution to a FastMCP tool server via a LangChain ReAct agent loop:

```bash
streamlit run app_mcp.py
```




