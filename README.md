# Advanced RAG System 🚀

A production-grade Retrieval-Augmented Generation (RAG) system designed for scalable enterprise knowledge retrieval, hybrid search, and local LLM-powered reasoning.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![LLM](https://img.shields.io/badge/LLM-Mistral%20%7C%20Llama3-orange)

## 🌟 Overview

This project demonstrates a modern RAG architecture with:
- Hybrid retrieval pipelines
- Re-ranking for retrieval optimization
- Local LLM inference
- Streaming responses
- Multi-format document ingestion
- Evaluation pipelines for answer quality and retrieval relevance

The goal is to simulate production-grade GenAI systems used in enterprise AI assistants, internal research copilots, and document intelligence platforms.

---

## 🏗️ System Architecture

```text
User Query
    ↓
Query Rewriting & Expansion
    ↓
Hybrid Retrieval (Dense + Sparse)
    ↓
Cross-Encoder Re-ranking
    ↓
Context Assembly
    ↓
LLM Generation
    ↓
Streaming Response + Citations
```

### Core Components

| Layer | Technologies |
|------|------|
| LLM Inference | Ollama, Llama.cpp |
| Embeddings | SentenceTransformers |
| Vector Store | FAISS / Qdrant / ChromaDB |
| Retrieval | BM25 + Dense Retrieval |
| Backend | FastAPI |
| Deployment | Docker |
| Evaluation | MRR, Recall@K, Faithfulness |

---

## 🚀 Key Features

### Advanced Retrieval
- Hybrid dense + sparse search
- Multi-query expansion
- HyDE retrieval
- Parent-child chunking
- Cross-encoder re-ranking

### Document Intelligence
- PDF, DOCX, TXT, Markdown ingestion
- Metadata extraction
- Semantic chunking
- Structured document parsing

### Production-Oriented Design
- Streaming responses
- Conversation memory
- Configurable retrieval pipeline
- Modular architecture
- Offline/local LLM support

### Evaluation & Benchmarking
- Retrieval evaluation metrics
- Relevance scoring
- Benchmark comparisons
- Pipeline experimentation support

---

## 📂 Project Structure

```text
advanced-rag-system/
├── backend/
├── frontend/
├── configs/
├── evaluation/
├── notebooks/
├── docker/
├── tests/
├── docs/
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Devmac21/advanced-rag-system.git
cd advanced-rag-system

pip install -r requirements.txt
```

### Run Locally

```bash
python app.py
```

---

## 🐳 Docker Deployment

```bash
docker build -t advanced-rag .
docker run -p 8000:8000 advanced-rag
```

---

## 📊 Future Improvements

- [ ] Multi-modal RAG support
- [ ] Graph-based retrieval
- [ ] Async inference pipelines
- [ ] GPU inference optimization
- [ ] Real-time indexing
- [ ] Distributed vector retrieval

---

## 📈 Why This Project Matters

This repository focuses on practical AI engineering concepts increasingly required for:
- AI Engineer roles
- ML Engineer positions
- Applied Scientist internships
- Research Engineering roles

The architecture reflects real-world patterns used in modern GenAI systems.

---

## 👨‍💻 Author

**Divanshu**  
AI/ML Engineer & Researcher  
Research Intern @ Carnegie Mellon University (CMU)  
Focused on LLMs, Multimodal AI, RAG Systems, and AI Infrastructure
