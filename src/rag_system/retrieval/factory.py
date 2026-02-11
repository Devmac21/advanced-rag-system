"""
Factory for creating retrievers.
"""

from typing import List

from ..config import RetrievalConfig
from ..embeddings.base import EmbeddingModel
from ..models import Chunk, RetrievalStrategy
from ..utils.logger import get_logger
from ..vector_stores.base import VectorStore
from .base import Retriever
from .dense_retriever import DenseRetriever
from .hybrid_retriever import HybridRetriever
from .sparse_retriever import SparseRetriever

logger = get_logger(__name__)


class RetrieverFactory:
    """Factory for creating retrievers."""
    
    @staticmethod
    def create(
        config: RetrievalConfig,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        chunks: List[Chunk],
    ) -> Retriever:
        """
        Create a retriever based on configuration.
        
        Args:
            config: Retrieval configuration
            vector_store: Vector store instance
            embedding_model: Embedding model instance
            chunks: List of all chunks (for sparse retrieval)
        
        Returns:
            Retriever instance
        """
        strategy = config.strategy
        
        logger.info(f"Creating retriever with strategy: {strategy}")
        
        if strategy == RetrievalStrategy.DENSE:
            return DenseRetriever(vector_store, embedding_model)
        
        elif strategy == RetrievalStrategy.SPARSE:
            return SparseRetriever(chunks, config)
        
        elif strategy == RetrievalStrategy.HYBRID:
            dense = DenseRetriever(vector_store, embedding_model)
            sparse = SparseRetriever(chunks, config)
            return HybridRetriever(dense, sparse, config)
        
        else:
            logger.warning(f"Unknown retrieval strategy: {strategy}, using dense")
            return DenseRetriever(vector_store, embedding_model)
