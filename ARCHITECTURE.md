# System Architecture

This document describes the architecture of the Advanced RAG System.

## Overview

The system follows a modular, production-ready architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    User Interfaces                       │
│         CLI │ Web UI │ Python API │ REST API            │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  RAG Pipeline                            │
│  (Orchestrates all components and manages workflow)     │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
┌───────▼──────┐ ┌──▼─────┐ ┌───▼────────┐
│  Ingestion   │ │Retrieval│ │Generation  │
│              │ │         │ │            │
│- Loaders     │ │- Dense  │ │- LLM       │
│- Chunkers    │ │- Sparse │ │- Prompts   │
│              │ │- Hybrid │ │- Streaming │
└──────┬───────┘ └──┬──────┘ └────────────┘
       │            │
┌──────▼────────────▼──────┐
│    Vector Store           │
│  (FAISS/ChromaDB/Qdrant) │
└───────────────────────────┘
```

## Core Components

### 1. Pipeline Layer (`pipeline.py`)

**Responsibilities:**
- Orchestrates all components
- Manages document ingestion workflow
- Handles query processing pipeline
- Conversation management
- Caching and performance optimization

**Key Features:**
- Stateful operation (maintains indexed documents)
- Automatic component initialization
- Error handling and recovery
- Performance metrics tracking

### 2. Ingestion Module (`ingestion/`)

#### Document Loaders (`loader.py`)
**Purpose:** Load documents from various sources

**Supported Formats:**
- PDF (via `pypdf`)
- DOCX (via `python-docx`)
- Text files
- Markdown
- HTML
- Code files (Python, JavaScript, etc.)

**Design Pattern:** Strategy Pattern
- `DocumentLoader` base class
- Specific loaders for each format
- `DocumentLoaderFactory` for automatic format detection

#### Text Chunkers (`chunker.py`)
**Purpose:** Split documents into optimal chunks

**Strategies:**
1. **Fixed Size:** Simple character-based splitting
2. **Recursive:** Splits at natural boundaries (paragraphs, sentences)
3. **Semantic:** Groups semantically similar sentences
4. **Parent-Child:** Hierarchical chunking for context preservation

**Design Pattern:** Strategy Pattern
- `TextChunker` base class
- Strategy-specific implementations
- `ChunkerFactory` for strategy selection

### 3. Embeddings Module (`embeddings/`)

**Purpose:** Convert text to vector representations

**Components:**
- `EmbeddingModel` base interface
- `SentenceTransformerEmbedding` implementation
- `EmbeddingFactory` for model creation

**Features:**
- Batch processing for efficiency
- GPU support
- Embedding normalization
- Model caching

**Extensibility:**
Can easily add:
- OpenAI embeddings
- Cohere embeddings
- Custom fine-tuned models

### 4. Vector Store Module (`vector_stores/`)

**Purpose:** Store and search vector embeddings

**Implementations:**
1. **FAISS** (`faiss_store.py`)
   - Fast similarity search
   - Multiple index types (Flat, IVF, HNSW)
   - Disk persistence
   - Best for: High performance, large datasets

2. **ChromaDB** (`chroma_store.py`)
   - Persistent by default
   - Metadata filtering
   - Easy to use
   - Best for: Development, small-medium datasets

**Design Pattern:** Abstract Factory
- `VectorStore` base interface
- Store-specific implementations
- `VectorStoreFactory` for instantiation

### 5. Retrieval Module (`retrieval/`)

**Purpose:** Find relevant documents for queries

**Components:**

#### Dense Retriever (`dense_retriever.py`)
- Uses embedding similarity
- Fast with vector stores
- Good for semantic matching

#### Sparse Retriever (`sparse_retriever.py`)
- BM25 algorithm
- Keyword-based matching
- Good for exact matches

#### Hybrid Retriever (`hybrid_retriever.py`)
- Combines dense + sparse
- Weighted score fusion
- Best overall performance

#### Re-ranker (`reranker.py`)
- Cross-encoder models
- Improves ranking quality
- Trades speed for accuracy

#### Query Processor (`query_processor.py`)
- Query expansion
- HyDE (Hypothetical Document Embeddings)
- Query rewriting

**Design Pattern:** Strategy + Composite
- `Retriever` base interface
- Multiple retrieval strategies
- Composition for hybrid approach

### 6. Generation Module (`generation/`)

**Purpose:** Generate answers using LLMs

**Components:**
- `LLM` base interface
- `OllamaLLM` for local inference
- `LLMFactory` for provider selection

**Features:**
- Streaming support
- Token counting
- Temperature control
- Stop sequences

**Extensibility:**
Easy to add:
- OpenAI GPT models
- Anthropic Claude
- HuggingFace models
- Custom models

### 7. Evaluation Module (`evaluation/`)

**Purpose:** Measure system performance

**Metrics:**

#### Retrieval Metrics
- Precision@K
- Recall@K
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)

#### Generation Metrics
- Faithfulness
- Relevance
- Coherence
- Answer quality

**Usage:**
```python
from rag_system.evaluation import RAGEvaluator

evaluator = RAGEvaluator(pipeline)
metrics = evaluator.run_full_evaluation(test_queries, ground_truth)
```

### 8. Configuration System (`config.py`)

**Purpose:** Centralized configuration management

**Features:**
- YAML file support
- Environment variable overrides
- Type validation (via Pydantic)
- Nested configuration
- Default values

**Design Pattern:** Builder Pattern

## Data Flow

### Ingestion Flow
```
Document File
    ↓
DocumentLoader (parse format)
    ↓
Document Object
    ↓
TextChunker (split into chunks)
    ↓
Chunk Objects
    ↓
EmbeddingModel (vectorize)
    ↓
Embeddings
    ↓
VectorStore (index)
```

### Query Flow
```
User Query
    ↓
QueryProcessor (expand/rewrite)
    ↓
Retriever (search)
    ↓
Retrieved Chunks
    ↓
Reranker (optional)
    ↓
Top K Chunks
    ↓
LLM (generate answer)
    ↓
Response with Sources
```

## Design Patterns

### 1. Strategy Pattern
- **Where:** Chunking, Retrieval, Embedding
- **Why:** Swap algorithms without changing client code
- **Benefit:** Easy experimentation with different approaches

### 2. Factory Pattern
- **Where:** All component creation
- **Why:** Centralize object creation logic
- **Benefit:** Easy to extend with new implementations

### 3. Pipeline Pattern
- **Where:** Query processing flow
- **Why:** Sequential processing stages
- **Benefit:** Clear, maintainable flow

### 4. Singleton Pattern (via caching)
- **Where:** Model loading
- **Why:** Expensive to load multiple times
- **Benefit:** Performance optimization

## Scalability Considerations

### Current Design
- Single-machine deployment
- In-memory vector stores (with persistence)
- Sequential processing

### Future Scaling Options

#### Horizontal Scaling
```
┌─────────────┐
│Load Balancer│
└──────┬──────┘
   ┌───┴───┬────────┐
   │       │        │
┌──▼──┐ ┌──▼──┐ ┌──▼──┐
│RAG 1│ │RAG 2│ │RAG 3│
└──┬──┘ └──┬──┘ └──┬──┘
   └───┬───┴────┬───┘
   ┌───▼────────▼───┐
   │ Shared Vector  │
   │     Store      │
   └────────────────┘
```

#### Distributed Processing
- Use Qdrant/Weaviate for distributed vector store
- Redis for caching layer
- Message queue for async processing
- Multiple LLM instances

## Security Considerations

### Current Implementation
- No authentication (local use)
- File system access control
- Input sanitization

### Production Recommendations
- Add API authentication (JWT)
- Implement rate limiting
- Sanitize user inputs
- Encrypt sensitive data
- Use secure LLM providers
- Implement access control for collections

## Performance Optimization

### Current Optimizations
1. **Batch Processing:** Embeddings generated in batches
2. **Caching:** Vector stores persisted to disk
3. **Lazy Loading:** Models loaded on-demand
4. **Streaming:** Responses streamed for better UX

### Future Optimizations
1. **Response Caching:** Cache common queries
2. **Async Processing:** Non-blocking operations
3. **GPU Acceleration:** Use CUDA for embeddings
4. **Index Optimization:** HNSW/IVF for large datasets
5. **Connection Pooling:** Reuse HTTP connections

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock external dependencies
- Fast execution

### Integration Tests
- Component interaction
- End-to-end workflows
- Requires running services

### Performance Tests
- Latency benchmarks
- Throughput testing
- Resource usage monitoring

## Extension Points

### Easy to Add:
1. **New Document Formats:** Implement `DocumentLoader`
2. **New Chunking Strategies:** Implement `TextChunker`
3. **New Retrieval Methods:** Implement `Retriever`
4. **New LLM Providers:** Implement `LLM`
5. **New Vector Stores:** Implement `VectorStore`

### Example: Adding OpenAI Embeddings
```python
class OpenAIEmbedding(EmbeddingModel):
    def embed_documents(self, texts):
        # Implementation
        pass
```

Update factory:
```python
if config.provider == "openai":
    return OpenAIEmbedding(config)
```

## Monitoring and Observability

### Current Logging
- Structured logging via `loguru`
- Configurable log levels
- File rotation

### Recommended Additions
- Metrics collection (Prometheus)
- Distributed tracing (OpenTelemetry)
- Error tracking (Sentry)
- Performance monitoring

## Deployment Options

### 1. Local Development
```bash
python cli.py serve
```

### 2. Docker Container
```bash
docker-compose up
```

### 3. Cloud Deployment
- AWS: ECS/EKS
- GCP: Cloud Run/GKE
- Azure: Container Instances/AKS

### 4. Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: advanced-rag
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: rag-system
        image: advanced-rag:latest
```

## Conclusion

This architecture provides:
- ✅ Modularity and maintainability
- ✅ Extensibility for new features
- ✅ Production-ready code quality
- ✅ Performance optimization
- ✅ Clear separation of concerns
- ✅ Easy testing and debugging

The system is designed to grow from a proof-of-concept to a production system with minimal refactoring.
