# Enterprise RAG Platform

An end-to-end **Retrieval-Augmented Generation (RAG)** platform that enables users to upload PDF and TXT documents and ask questions based on their contents. The system combines semantic search, BM25 keyword retrieval, FAISS vector search and locally hosted LLMs to generate context-aware answers.

## Overview

Large Language Models (LLMs) can generate incorrect or outdated information when they lack access to domain-specific knowledge.

This project addresses that limitation by implementing a document-based RAG architecture that retrieves relevant information from user-provided documents and uses that context to generate answers.

The platform supports dynamic document ingestion, automatic indexing and question answering through REST APIs.

## Key Features

- **Document Ingestion:** Upload PDF and TXT documents through FastAPI.
- **Text Extraction:** Extract text from PDF and TXT files.
- **Document Chunking:** Split extracted content into smaller, manageable chunks.
- **Semantic Embeddings:** Generate vector representations using Sentence Transformers.
- **Vector Search:** Store and retrieve document embeddings using FAISS.
- **Keyword Search:** Retrieve relevant information using BM25.
- **Hybrid Retrieval:** Combine semantic and keyword-based retrieval.
- **Context Building:** Prepare retrieved document chunks as context for the LLM.
- **LLM Integration:** Generate context-aware responses using Ollama.
- **Source Attribution:** Include retrieved source information in responses.
- **REST API:** Expose document upload, health-check and question-answering endpoints.

## Architecture

```text
                 User
                   |
                   v
              FastAPI
                   |
          +--------+--------+
          |                 |
          v                 v
     Upload API          Ask API
          |                 |
          v                 v
   Document Loader     User Question
          |                 |
          v                 v
    Text Extraction    Query Embedding
          |                 |
          v                 v
   Document Chunking   Hybrid Retrieval
          |             /          \
          v            v            v
    Embeddings      FAISS         BM25
          |            \            /
          v             \          /
    Vector Storage       Retrieved Chunks
          |                    |
          v                    v
     Index Update         Context Builder
                               |
                               v
                         Prompt Template
                               |
                               v
                         Ollama LLM
                               |
                               v
                         Generated Answer
                               |
                               v
                             User
```

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend Framework | FastAPI |
| Embedding Model | Sentence Transformers |
| Embedding Architecture | all-MiniLM-L6-v2 |
| Vector Database / Search | FAISS |
| Keyword Retrieval | BM25 |
| Numerical Computing | NumPy |
| LLM Runtime | Ollama |
| LLM Model | Llama 3.2 (3B) |
| Document Processing | PyPDF |
| API Testing | Swagger UI |
| Testing | Python unittest |

## Project Structure

```text
enterprise-rag-platform/
│
├── app/
│   ├── main.py
│   │
│   └── rag/
│       ├── bm25_retriever.py
│       ├── chunker.py
│       ├── context_builder.py
│       ├── document_loader.py
│       ├── embedding_service.py
│       ├── faiss_vector_store.py
│       ├── generation_service.py
│       ├── hybrid_retriever.py
│       ├── indexing_service.py
│       ├── models.py
│       ├── prompt_template.py
│       ├── rag_pipeline.py
│       ├── retriever.py
│       └── vector_store.py
│
├── data/
│   ├── evaluation/
│   ├── index/
│   ├── research_papers/
│   └── uploads/
│
├── scripts/
│   ├── benchmark_faiss.py
│   ├── download_research_papers.py
│   ├── evaluate_retrieval.py
│   ├── index_documents.py
│   └── inspect_chunks.py
│
├── tests/
│   ├── test_bm25.py
│   ├── test_generation_service.py
│   ├── test_hybrid.py
│   └── test_rag_pipeline_unit.py
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## How It Works

### 1. Document Ingestion

Users upload PDF or TXT documents through the FastAPI document upload endpoint.

The document loader extracts the text and preserves relevant document metadata, including source information and PDF page numbers.

### 2. Text Chunking

Extracted content is divided into smaller chunks to improve retrieval relevance and provide manageable context to the language model.

### 3. Embedding Generation

The Sentence Transformers model `all-MiniLM-L6-v2` converts document chunks into 384-dimensional vector embeddings.

### 4. Indexing and Storage

The generated embeddings and associated chunk metadata are stored using FAISS and a JSON-based metadata store.

### 5. Hybrid Retrieval

When a user submits a question, the system uses:

- **Semantic Retrieval:** Finds contextually similar document chunks using vector embeddings and FAISS.
- **Keyword Retrieval:** Uses BM25 to identify relevant chunks based on matching terms.
- **Hybrid Retrieval:** Combines both retrieval approaches using Reciprocal Rank Fusion (RRF).

### 6. Context Construction

Retrieved chunks are assembled into a context containing relevant source information.

### 7. Response Generation

The system constructs a prompt using the user's question and retrieved context. Ollama runs the locally hosted Llama 3.2 model to generate the final response.

## Installation and Setup

### Prerequisites

- Python 3.12
- Git
- Ollama
- Llama 3.2 (3B) model

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-rag-platform.git

cd enterprise-rag-platform
```

### 2. Create a Virtual Environment

Windows:

```powershell
py -3.12 -m venv .venv

.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file using `.env.example` as a reference.

Configure the Ollama settings:

```env
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
```

### 5. Install and Start Ollama

Download Ollama from:

https://ollama.com/

Pull the required model:

```bash
ollama pull llama3.2:3b
```

Ensure Ollama is running locally.

### 6. Start the FastAPI Application

```bash
uvicorn app.main:app --reload
```

The application will be available at:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check application health |
| POST | `/documents/upload` | Upload and index PDF/TXT documents |
| POST | `/ask` | Ask questions using indexed documents |

Use Swagger UI (`/docs`) to interact with and test the API endpoints.

## Running Tests

The project uses Python's built-in `unittest` framework.

Run the complete test suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The test suite covers:

- BM25 retrieval.
- Generation service.
- Hybrid retrieval.
- RAG pipeline.
- Input validation and fallback handling.

**Current test status:** 17 tests passing.

## Evaluation and Benchmarking

The project includes utility scripts for retrieval evaluation and FAISS benchmarking.

Run retrieval evaluation:

```bash
python -m scripts.evaluate_retrieval
```

Run FAISS benchmarking:

```bash
python -m scripts.benchmark_faiss
```

Inspect indexed document chunks:

```bash
python -m scripts.inspect_chunks
```

## Design Considerations

- Local embedding generation reduces dependency on paid embedding APIs.
- Ollama enables local LLM inference.
- Hybrid retrieval combines semantic relevance with keyword matching.
- Document metadata supports source attribution.
- Modular architecture separates ingestion, retrieval, context construction and generation.
- FastAPI provides an interface for integrating the RAG pipeline with other applications.

## Future Scope

- Improved retrieval evaluation and relevance optimization.
- Advanced document processing and metadata filtering.
- Authentication and access control.
- Conversation history and multi-turn question answering.
- Containerization and cloud deployment.
- Monitoring, logging and performance optimization.

## Author

**Bhanu Prakash**

M.Tech – Artificial Intelligence and Machine Learning  
BITS Pilani (WILP)

Bachelor of Engineering – Electronics and Communication Engineering

---

**If you find this project useful, consider giving the repository a star!**