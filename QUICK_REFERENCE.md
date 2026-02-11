# Quick Reference Guide

## 🚀 Common Commands

### Installation
```bash
# Setup
python -m venv venv
venv\Scripts\activate                    # Windows
source venv/bin/activate                 # Linux/Mac
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Install Ollama
# Windows: Download from ollama.ai
# Linux: curl https://ollama.ai/install.sh | sh
ollama pull llama3.1
ollama serve
```

### Document Ingestion
```bash
# Single file
python cli.py ingest document.pdf

# Directory
python cli.py ingest ./documents --recursive

# Specific collection
python cli.py ingest ./docs --collection my_docs

# With custom config
python cli.py ingest ./docs --config-file configs/custom.yaml
```

### Querying
```bash
# One-off query
python cli.py query "What is machine learning?"

# With more results
python cli.py query "Explain deep learning" --top-k 10

# Interactive chat
python cli.py chat

# With specific collection
python cli.py chat --collection my_docs
```

### Web Interface
```bash
# Start server
python cli.py serve

# Custom port
python cli.py serve --port 8080

# Access
# http://localhost:8501
```

### Docker
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Collection Management
```bash
# Clear collection
python cli.py clear --collection my_docs

# List stats
python cli.py query "test" --collection my_docs
# (Stats shown after query)
```

## 🐍 Python API

### Basic Usage
```python
from src.rag_system import Config, RAGPipeline

# Initialize
config = Config.from_yaml("configs/default.yaml")
pipeline = RAGPipeline(config)

# Ingest
pipeline.ingest_directory("./documents")

# Query
response = pipeline.query("What is Python?")
print(response.answer)
```

### Streaming
```python
for chunk in pipeline.stream_query("Explain AI"):
    print(chunk, end="", flush=True)
```

### Conversations
```python
conv_id = "user123"
r1 = pipeline.query("What is Python?", conversation_id=conv_id)
r2 = pipeline.query("What are its features?", conversation_id=conv_id)
```

### Custom Config
```python
from src.rag_system.models import ChunkingStrategy, RetrievalStrategy

config = Config()
config.chunking.strategy = ChunkingStrategy.SEMANTIC
config.retrieval.strategy = RetrievalStrategy.HYBRID
config.retrieval.enable_reranking = True
config.llm.temperature = 0.5

pipeline = RAGPipeline(config)
```

## ⚙️ Configuration Options

### Chunking Strategies
- `fixed` - Simple character-based splitting
- `recursive` - Split at natural boundaries (best default)
- `semantic` - Group similar sentences
- `parent_child` - Hierarchical for context

### Retrieval Strategies
- `dense` - Semantic search only
- `sparse` - Keyword search (BM25)
- `hybrid` - Both combined (recommended)

### Vector Stores
- `faiss` - Fast, in-memory (with persistence)
- `chromadb` - Easy, persistent
- `qdrant` - Production-grade (coming soon)

### LLM Providers
- `ollama` - Local models (default)
- `openai` - GPT models (add API key)
- `anthropic` - Claude (add API key)

## 📝 Config File Template

```yaml
# configs/my_config.yaml

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"  # or "cuda"

chunking:
  strategy: "recursive"
  chunk_size: 512
  chunk_overlap: 50

vector_store:
  type: "faiss"
  collection_name: "my_docs"

retrieval:
  strategy: "hybrid"
  top_k: 5
  enable_reranking: true

llm:
  provider: "ollama"
  model: "llama3.1"
  temperature: 0.7
```

## 🔧 Environment Variables

```bash
# .env file
RAG_LOG_LEVEL=INFO
RAG_LLM__MODEL=llama3.1
RAG_RETRIEVAL__TOP_K=10
RAG_EMBEDDINGS__DEVICE=cuda
RAG_VECTOR_STORE__TYPE=chromadb
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/rag_system

# Run example
python examples/quickstart.py
```

## 📊 Performance Tips

### Speed Up
```yaml
embeddings:
  device: "cuda"  # Use GPU
  batch_size: 64   # Larger batches

vector_store:
  index_type: "HNSW"  # Faster search
```

### Better Quality
```yaml
retrieval:
  strategy: "hybrid"
  enable_reranking: true
  top_k: 10

chunking:
  strategy: "parent_child"
```

### Balance
```yaml
retrieval:
  strategy: "hybrid"
  dense_weight: 0.7
  sparse_weight: 0.3
  enable_reranking: true
  rerank_top_k: 20
```

## 🐛 Troubleshooting

### "No module named 'rag_system'"
```bash
# Ensure you're in project root
cd RAG
# Activate virtual environment
venv\Scripts\activate
```

### "Connection refused to Ollama"
```bash
# Start Ollama server
ollama serve
```

### "CUDA out of memory"
```yaml
# Use CPU in config
embeddings:
  device: "cpu"
```

### Slow performance
```bash
# Use smaller model
ollama pull phi
# Update config
llm:
  model: "phi"
```

## 📚 File Locations

| Item | Path |
|------|------|
| Configuration | `configs/default.yaml` |
| Vector stores | `./vector_stores/` |
| Logs | `./logs/` |
| Cache | `./cache/` |
| Examples | `examples/` |
| Tests | `tests/` |

## 🎯 Common Workflows

### New Project Setup
```bash
# 1. Install
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Start Ollama
ollama serve
ollama pull llama3.1

# 3. Test
python examples/quickstart.py
```

### Ingest Company Docs
```bash
# 1. Ingest
python cli.py ingest ./company_docs --collection company

# 2. Chat
python cli.py chat --collection company

# 3. Or Web UI
python cli.py serve --collection company
```

### Experiment with Settings
```bash
# 1. Copy config
cp configs/default.yaml configs/experiment.yaml

# 2. Edit settings
# vim configs/experiment.yaml

# 3. Test
python cli.py chat --config-file configs/experiment.yaml
```

### Deploy with Docker
```bash
# 1. Build
docker-compose build

# 2. Run
docker-compose up -d

# 3. Access
# http://localhost:8501
```

## 💡 Pro Tips

1. **Start Simple:** Use default config, basic chunking, then optimize

2. **Experiment:** Try different chunking sizes and retrieval strategies

3. **Monitor:** Check response times and quality, adjust accordingly

4. **Cache:** Enable caching for frequently used queries

5. **GPU:** Use CUDA for faster embeddings if available

6. **Batch Ingest:** Process documents in batches for large collections

7. **Test First:** Use quickstart.py to verify setup before custom work

## 🔗 Useful Links

- [Full Documentation](USAGE.md)
- [Installation Guide](INSTALL.md)
- [Architecture](ARCHITECTURE.md)
- [Contributing](CONTRIBUTING.md)
- [Ollama Models](https://ollama.ai/library)
- [Sentence Transformers](https://www.sbert.net/)

---

**Quick Help:** `python cli.py --help`

For detailed documentation, see [USAGE.md](USAGE.md)
