"""
Data models for the RAG system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class ChunkingStrategy(str, Enum):
    """Supported chunking strategies."""
    FIXED = "fixed"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    PARENT_CHILD = "parent_child"


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    FAISS = "faiss"
    CHROMA = "chromadb"
    QDRANT = "qdrant"


class RetrievalStrategy(str, Enum):
    """Supported retrieval strategies."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"


@dataclass
class Document:
    """Represents a source document."""
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Ensure metadata contains essential fields."""
        if "source" not in self.metadata and self.source:
            self.metadata["source"] = self.source


@dataclass
class Chunk:
    """Represents a text chunk from a document."""
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    
    # Chunk position info
    start_idx: int = 0
    end_idx: int = 0
    chunk_index: int = 0
    
    def __post_init__(self):
        """Ensure metadata consistency."""
        if self.document_id and "document_id" not in self.metadata:
            self.metadata["document_id"] = self.document_id


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with relevance score."""
    chunk: Chunk
    score: float
    rank: int = 0
    
    def __lt__(self, other):
        """For sorting by score (descending)."""
        return self.score > other.score


@dataclass
class QueryResponse:
    """Response from RAG query."""
    query: str
    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def get_source_documents(self) -> List[str]:
        """Get list of unique source documents."""
        sources = set()
        for retrieved in self.sources:
            if "source" in retrieved.chunk.metadata:
                sources.add(retrieved.chunk.metadata["source"])
        return list(sources)
    
    def format_sources(self) -> str:
        """Format sources for display."""
        if not self.sources:
            return "No sources found."
        
        output = []
        for i, retrieved in enumerate(self.sources[:5], 1):
            source = retrieved.chunk.metadata.get("source", "Unknown")
            score = retrieved.score
            preview = retrieved.chunk.content[:200].replace("\n", " ")
            output.append(f"{i}. {source} (score: {score:.3f})\n   {preview}...")
        
        return "\n\n".join(output)


@dataclass
class EvaluationMetrics:
    """Metrics for RAG evaluation."""
    # Retrieval metrics
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    
    # Answer quality metrics
    faithfulness: float = 0.0
    relevance: float = 0.0
    coherence: float = 0.0
    
    # Performance metrics
    avg_retrieval_time: float = 0.0
    avg_generation_time: float = 0.0
    avg_total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "retrieval": {
                "precision_at_k": self.precision_at_k,
                "recall_at_k": self.recall_at_k,
                "mrr": self.mrr,
                "ndcg": self.ndcg,
            },
            "answer_quality": {
                "faithfulness": self.faithfulness,
                "relevance": self.relevance,
                "coherence": self.coherence,
            },
            "performance": {
                "avg_retrieval_time": self.avg_retrieval_time,
                "avg_generation_time": self.avg_generation_time,
                "avg_total_time": self.avg_total_time,
            }
        }


@dataclass
class ConversationMessage:
    """Message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """Represents a conversation with context."""
    id: str = field(default_factory=lambda: str(uuid4()))
    messages: List[ConversationMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation."""
        msg = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(msg)
        self.updated_at = datetime.now()
    
    def get_context(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation context."""
        recent = self.messages[-max_messages:]
        return [{"role": msg.role, "content": msg.content} for msg in recent]
