"""
Dense retrieval using embeddings.
"""

from typing import List

from ..embeddings.base import EmbeddingModel
from ..models import RetrievedChunk
from ..utils.logger import get_logger
from ..vector_stores.base import VectorStore
from .base import Retriever

logger = get_logger(__name__)


class DenseRetriever(Retriever):
    """Dense retrieval using vector similarity."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
    ):
        """
        Initialize dense retriever.
        
        Args:
            vector_store: Vector store instance
            embedding_model: Embedding model instance
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Retrieve using dense embeddings."""
        # Embed query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # Convert to RetrievedChunk objects
        retrieved = []
        for rank, (chunk, score) in enumerate(results):
            retrieved_chunk = RetrievedChunk(
                chunk=chunk,
                score=score,
                rank=rank,
            )
            retrieved.append(retrieved_chunk)
        
        logger.debug(f"Dense retrieval found {len(retrieved)} chunks")
        return retrieved
