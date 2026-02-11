# Usage Guide

Complete guide for using the Advanced RAG System.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Command Line Interface](#command-line-interface)
3. [Web Interface](#web-interface)
4. [Python API](#python-api)
5. [Configuration](#configuration)
6. [Advanced Features](#advanced-features)

## Quick Start

### 1. Ingest Documents

```bash
# Ingest a single file
python cli.py ingest document.pdf

# Ingest a directory
python cli.py ingest ./documents --recursive

# Ingest to a specific collection
python cli.py ingest ./documents --collection my_docs
```

### 2. Ask Questions

```bash
# One-off question
python cli.py query "What is machine learning?" --collection my_docs

# Interactive chat
python cli.py chat --collection my_docs
```

### 3. Launch Web UI

```bash
python cli.py serve --port 8000
```

Visit http://localhost:8000 in your browser.

## Command Line Interface

### Available Commands

#### `ingest`
Ingest documents into the system.

```bash
python cli.py ingest <source> [options]

Options:
  --collection TEXT      Collection name (default: "default")
  --config-file TEXT     Path to config YAML file
  --recursive            Recursively search directories (default: True)
```

Examples:
```bash
# Ingest PDF
python cli.py ingest research_paper.pdf

# Ingest directory with custom config
python cli.py ingest ./docs --config-file configs/custom.yaml

# Ingest to specific collection
python cli.py ingest ./legal_docs --collection legal
```

#### `query`
Ask a single question.

```bash
python cli.py query <question> [options]

Options:
  --collection TEXT      Collection name
  --config-file TEXT     Config file path
  --top-k INTEGER        Number of chunks to retrieve (default: 5)
  --stream              Stream the response
```

Examples:
```bash
# Basic query
python cli.py query "What are the main findings?"

# With streaming
python cli.py query "Explain the methodology" --stream

# Retrieve more chunks
python cli.py query "Summary of results" --top-k 10
```

#### `chat`
Start interactive chat session.

```bash
python cli.py chat [options]

Options:
  --collection TEXT      Collection name
  --config-file TEXT     Config file path
```

Example session:
```bash
$ python cli.py chat

You: What is Python?
Assistant: Python is a high-level programming language...

You: What are its main features?
Assistant: Python's main features include...

You: exit
```

#### `serve`
Start web UI server.

```bash
python cli.py serve [options]

Options:
  --host TEXT            Host to bind (default: "127.0.0.1")
  --port INTEGER         Port to bind (default: 8000)
  --collection TEXT      Collection name
  --config-file TEXT     Config file path
```

#### `clear`
Clear a collection.

```bash
python cli.py clear --collection my_docs
```

## Web Interface

### Features

1. **Chat Interface**
   - Ask questions and get answers
   - View sources with confidence scores
   - Conversation history
   - Streaming responses

2. **Document Ingestion**
   - Upload files via web interface
   - Ingest local directories
   - Progress tracking

3. **Statistics Dashboard**
   - View system metrics
   - Configuration details
   - Collection management

### Usage

1. Start the server:
   ```bash
   python cli.py serve --port 8501
   ```

2. Open browser to http://localhost:8501

3. Navigate through tabs:
   - **Chat**: Ask questions
   - **Ingest**: Upload documents
   - **Statistics**: View system stats

## Python API

### Basic Usage

```python
from src.rag_system import Config, RAGPipeline

# Initialize
config = Config.from_yaml("configs/default.yaml")
pipeline = RAGPipeline(config)

# Ingest documents
pipeline.ingest_directory("./documents")

# Query
response = pipeline.query("What is machine learning?")
print(response.answer)
print(f"Sources: {response.get_source_documents()}")
```

### Advanced Usage

#### Streaming Responses

```python
for chunk in pipeline.stream_query("Explain neural networks"):
    print(chunk, end="", flush=True)
```

#### Multi-turn Conversations

```python
conversation_id = "user_123"

# First question
response1 = pipeline.query(
    "What is Python?",
    conversation_id=conversation_id
)

# Follow-up question (with context)
response2 = pipeline.query(
    "What are its main features?",
    conversation_id=conversation_id
)
```

#### Custom Configuration

```python
from src.rag_system import Config
from src.rag_system.models import ChunkingStrategy, RetrievalStrategy

config = Config()

# Customize chunking
config.chunking.strategy = ChunkingStrategy.SEMANTIC
config.chunking.chunk_size = 1024

# Customize retrieval
config.retrieval.strategy = RetrievalStrategy.HYBRID
config.retrieval.top_k = 10
config.retrieval.enable_reranking = True

# Customize LLM
config.llm.model = "mistral"
config.llm.temperature = 0.5

pipeline = RAGPipeline(config)
```

## Configuration

### Config File Structure

```yaml
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"  # or "cuda"

chunking:
  strategy: "recursive"  # fixed, semantic, parent_child
  chunk_size: 512
  chunk_overlap: 50

vector_store:
  type: "faiss"  # or "chromadb"
  persist_directory: "./vector_stores"

retrieval:
  strategy: "hybrid"  # dense, sparse, hybrid
  top_k: 5
  enable_reranking: true

llm:
  provider: "ollama"
  model: "llama3.1"
  temperature: 0.7
```

### Environment Variables

Set via `.env` file or environment:

```bash
RAG_LOG_LEVEL=INFO
RAG_LLM__MODEL=llama3.1
RAG_RETRIEVAL__TOP_K=10
```

## Advanced Features

### 1. Hybrid Search

Combines dense (embedding) and sparse (BM25) retrieval:

```yaml
retrieval:
  strategy: "hybrid"
  dense_weight: 0.7
  sparse_weight: 0.3
```

### 2. Re-ranking

Improves relevance with cross-encoder:

```yaml
retrieval:
  enable_reranking: true
  rerank_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  rerank_top_k: 20  # Retrieve more, then re-rank
```

### 3. Query Expansion

Generate multiple query variations:

```yaml
retrieval:
  enable_query_expansion: true
  num_expanded_queries: 3
```

### 4. HyDE (Hypothetical Document Embeddings)

Generate hypothetical answers for better retrieval:

```yaml
retrieval:
  enable_hyde: true
```

### 5. Parent-Child Chunking

Retrieve precise chunks, use larger context:

```yaml
chunking:
  strategy: "parent_child"
  parent_chunk_size: 2048
  child_chunk_size: 512
```

### 6. Multiple Collections

Organize documents into separate collections:

```python
# Ingest to different collections
pipeline.config.vector_store.collection_name = "technical"
pipeline.ingest_directory("./tech_docs")

pipeline.config.vector_store.collection_name = "legal"
pipeline.ingest_directory("./legal_docs")

# Query specific collection
response = pipeline.query("What is the license?")
```

## Performance Tips

1. **Use GPU for embeddings**:
   ```yaml
   embeddings:
     device: "cuda"
   ```

2. **Adjust chunk size** based on your documents:
   - Smaller chunks (256-512): Better for precise Q&A
   - Larger chunks (1024-2048): Better for summarization

3. **Enable caching**:
   ```yaml
   cache_enabled: true
   ```

4. **Tune retrieval parameters**:
   - Increase `top_k` for better recall
   - Enable re-ranking for better precision
   - Adjust `dense_weight` and `sparse_weight` for hybrid search

## Troubleshooting

### Slow Performance
- Enable GPU: `device: "cuda"`
- Reduce batch size: `batch_size: 16`
- Use HNSW index: `index_type: "HNSW"`

### Poor Answer Quality
- Increase `top_k`
- Enable re-ranking
- Try different chunking strategies
- Adjust chunk size

### Out of Memory
- Reduce batch size
- Use smaller embedding model
- Switch to CPU: `device: "cpu"`

## Next Steps

- Explore [examples/](examples/)
- Read [API Documentation](docs/API.md)
- Join the community
- Contribute!

For more help, see [INSTALL.md](INSTALL.md) or open an issue.
