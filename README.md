# RAG Chatbot API

A FastAPI-based Retrieval-Augmented Generation (RAG) chatbot that intelligently decides whether to answer a user query directly using an LLM or retrieve relevant information from internal company documents using embeddings and vector search.

---

## Architecture Overview

### High-Level Flow

Client → FastAPI → Assistant Agent → (LLM or RAG Tool) → Response

```
Client
  |
  | POST /bot/ask
  v
FastAPI (main.py)
  |
  v
Assistant Agent (decider.py)
  |
  |-- General knowledge → Azure OpenAI (Chat Completion)
  |
  |-- Company-specific query
         |
         v
     RAG Tool (tool.py)
         |
         v
     RAG Manager (rag_manager.py)
         |
         |-- PDF ingestion & chunking
         |-- Embeddings (Azure OpenAI)
         |-- FAISS vector search
         |
         v
     Azure OpenAI (Answer Generation)
```

---

## Tech Stack Used

### Backend
- Python 3.11+
- FastAPI
- Uvicorn

### AI / LLM
- Azure OpenAI (GPT-4o)
- Azure OpenAI Embeddings (text-embedding-3-large)
- AutoGen AgentChat

### Retrieval
- FAISS (local vector store)
- PyPDF2 (PDF parsing)
- NumPy

### Utilities & Security
- python-dotenv
- NanoID
- Custom X-Auth middleware

---

## Setup Instructions

### Local Setup

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

#### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Environment Variables

Create a `.env` file in the project root:

```env
# Chat Model
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=api_version
AZURE_OPENAI_MODEL_NAME=model_name
AZURE_OPENAI_DEPLOYMENT=model_name

# Embeddings (RAG)
AZURE_OPENAI_API_KEY2=your_key
AZURE_OPENAI_ENDPOINT2=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION2=api_version
AZURE_EMBEDDING_DEPLOYMENT2=model_name

# Security
X_AUTH_TOKEN=your-secret-token
```

#### 5. Run the Application
```bash
uvicorn main:app --reload
```

Access:
- API Root: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs

---

## Azure Deployment

1. Create an Azure OpenAI resource
2. Deploy:
   - GPT-4o for chat
   - text-embedding-3-large for embeddings
3. Create an Azure App Service (Linux + Python)
4. Set environment variables in App Service → Configuration
5. Deploy via GitHub Actions, Zip Deploy, or Azure CLI
6. Use Azure File Share or Blob Storage for:
   - documents/
   - embeddings/
   - metadata/

---

## API Endpoints

### Chatbot
- **POST /bot/ask**
  - Form fields:
    - `query` (required)
    - `session_id` (optional)

### RAG Management
- **POST /rag/documents/add** – Upload PDF
- **DELETE /rag/documents/delete/{doc_id}**
- **GET /rag/documents/get**
- **POST /rag/query**

🔐 All `/bot` and `/rag` routes are protected using `X-Auth-Token`.

---

## Design Decisions

- Agent-based routing using AutoGen to decide between RAG and direct LLM responses
- Tool abstraction for RAG to keep agent logic clean and extensible
- Local FAISS vector store for fast and lightweight retrieval
- Chunk-based PDF indexing to balance recall and context size
- Session-based conversation memory with bounded history
- Separation of chat and embedding models for flexibility and cost control

---

## Limitations

- In-memory session storage (not horizontally scalable)
- Local FAISS index (single-node only)
- PDF-only document support
- No document versioning
- No streaming responses
- Single-tenant design

---

## Future Improvements

- Redis or database-backed session memory
- Azure AI Search / Pinecone / Weaviate for vector storage
- Streaming responses from LLM
- Support for DOCX, TXT, HTML
- Role-based access control (RBAC)
- Multi-tenant support
- Frontend UI for chat and document management

---

## Summary

This project implements a production-grade RAG chatbot with intelligent agent routing, secure APIs, Azure OpenAI integration, and a modular architecture suitable for enterprise internal knowledge assistants.
