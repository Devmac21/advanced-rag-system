"""
Re-ranking module using cross-encoders.
"""

from typing import List

from sentence_transformers import CrossEncoder

from ..config import RetrievalConfig
from ..models import RetrievedChunk
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """Re-rank retrieved chunks using cross-encoder."""
    
    def __init__(self, config: RetrievalConfig):
        """
        Initialize reranker.
        
        Args:
            config: Retrieval configuration
        """
        self.config = config
        
        if config.enable_reranking:
            logger.info(f"Loading reranker model: {config.rerank_model}")
            self.model = CrossEncoder(config.rerank_model)
        else:
            self.model = None
    
    def rerank(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Re-rank retrieved chunks.
        
        Args:
            query: Query text
            retrieved_chunks: List of retrieved chunks
            top_k: Number of top results to return
        
        Returns:
            Re-ranked chunks
        """
        if not self.model or not retrieved_chunks:
            return retrieved_chunks[:top_k]
        
        # Prepare pairs for cross-encoder
        pairs = [(query, chunk.chunk.content) for chunk in retrieved_chunks]
        
        # Get scores from cross-encoder
        scores = self.model.predict(pairs)
        
        # Update scores
        for i, score in enumerate(scores):
            retrieved_chunks[i].score = float(score)
        
        # Sort by new scores
        reranked = sorted(retrieved_chunks, key=lambda x: x.score, reverse=True)[:top_k]
        
        # Update ranks
        for rank, chunk in enumerate(reranked):
            chunk.rank = rank
        
        logger.debug(f"Re-ranked {len(retrieved_chunks)} chunks to top {len(reranked)}")
        return reranked
