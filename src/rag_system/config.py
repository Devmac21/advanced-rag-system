"""
Configuration management for the RAG system.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

from .models import ChunkingStrategy, LLMProvider, RetrievalStrategy, VectorStoreType


class EmbeddingConfig(BaseSettings):
    """Configuration for embeddings."""
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True
    cache_dir: Optional[str] = None


class ChunkingConfig(BaseSettings):
    """Configuration for text chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Semantic chunking options
    use_semantic_similarity: bool = False
    similarity_threshold: float = 0.5
    
    # Parent-child options
    parent_chunk_size: int = 2048
    child_chunk_size: int = 512


class VectorStoreConfig(BaseSettings):
    """Configuration for vector store."""
    type: VectorStoreType = VectorStoreType.FAISS
    persist_directory: str = "./vector_stores"
    collection_name: str = "default"
    
    # FAISS specific
    index_type: str = "Flat"  # Flat, IVF, HNSW
    
    # ChromaDB specific
    chroma_host: Optional[str] = None
    chroma_port: Optional[int] = None
    
    # Qdrant specific
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None


class RetrievalConfig(BaseSettings):
    """Configuration for retrieval."""
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 5
    
    # Dense retrieval
    dense_weight: float = 0.7
    
    # Sparse retrieval (BM25)
    sparse_weight: float = 0.3
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    
    # Re-ranking
    enable_reranking: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 20  # Retrieve more, then re-rank to top_k
    
    # Query processing
    enable_query_expansion: bool = True
    enable_hyde: bool = False
    num_expanded_queries: int = 3


class LLMConfig(BaseSettings):
    """Configuration for LLM."""
    provider: str = "ollama"  # Allow any string for flexibility
    model: str = "llama3.1"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    
    # Ollama specific
    ollama_base_url: str = "http://localhost:11434"
    
    # Groq specific
    groq_api_key: Optional[str] = None
    
    # OpenAI specific
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    
    # Anthropic specific
    anthropic_api_key: Optional[str] = None
    
    # Streaming
    stream: bool = True
    
    # System prompt
    system_prompt: str = """You are a helpful AI assistant. Answer questions based on the provided context.
Be concise, accurate, and cite your sources. If you don't know the answer, say so."""


class PromptConfig(BaseSettings):
    """Configuration for prompts."""
    
    qa_template: str = """Context information is below:
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Question: {query}
Answer:"""
    
    hyde_template: str = """Write a detailed paragraph that would answer the following question:

Question: {query}

Paragraph:"""
    
    query_expansion_template: str = """Generate {num_queries} different versions of the following question to retrieve relevant documents:

Original question: {query}

Alternative questions:"""


class Config(BaseSettings):
    """Main configuration class."""
    
    # Component configurations
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)
    
    # General settings
    log_level: str = "INFO"
    cache_enabled: bool = True
    cache_dir: str = "./cache"
    
    # Conversation settings
    max_conversation_history: int = 10
    
    class Config:
        env_prefix = "RAG_"
        env_nested_delimiter = "__"
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        config_dict = self.model_dump()
        
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
