"""
Hybrid retrieval combining dense and sparse methods.
"""

from typing import Dict, List

from ..config import RetrievalConfig
from ..models import Chunk, RetrievedChunk
from ..utils.logger import get_logger
from .base import Retriever
from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever

logger = get_logger(__name__)


class HybridRetriever(Retriever):
    """Hybrid retrieval combining dense and sparse methods."""
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        config: RetrievalConfig,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_retriever: Dense retriever instance
            sparse_retriever: Sparse retriever instance
            config: Retrieval configuration
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.config = config
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Retrieve using hybrid approach."""
        # Retrieve from both methods
        # Get more results for fusion
        retrieval_k = min(top_k * 2, 20)
        
        dense_results = self.dense_retriever.retrieve(query, top_k=retrieval_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=retrieval_k)
        
        # Combine results using weighted scores
        combined = self._combine_results(dense_results, sparse_results)
        
        # Sort by combined score and take top k
        combined_sorted = sorted(combined, key=lambda x: x.score, reverse=True)[:top_k]
        
        # Update ranks
        for rank, retrieved in enumerate(combined_sorted):
            retrieved.rank = rank
        
        logger.debug(f"Hybrid retrieval combined {len(dense_results)} dense + {len(sparse_results)} sparse → {len(combined_sorted)} results")
        return combined_sorted
    
    def _combine_results(
        self,
        dense_results: List[RetrievedChunk],
        sparse_results: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """Combine dense and sparse results with weighted scoring."""
        # Create a mapping of chunk_id to RetrievedChunk
        chunk_scores: Dict[str, RetrievedChunk] = {}
        
        # Add dense results
        for retrieved in dense_results:
            chunk_id = retrieved.chunk.id
            weighted_score = retrieved.score * self.config.dense_weight
            
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id].score += weighted_score
            else:
                chunk_scores[chunk_id] = RetrievedChunk(
                    chunk=retrieved.chunk,
                    score=weighted_score,
                    rank=0,
                )
        
        # Add sparse results
        for retrieved in sparse_results:
            chunk_id = retrieved.chunk.id
            weighted_score = retrieved.score * self.config.sparse_weight
            
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id].score += weighted_score
            else:
                chunk_scores[chunk_id] = RetrievedChunk(
                    chunk=retrieved.chunk,
                    score=weighted_score,
                    rank=0,
                )
        
        return list(chunk_scores.values())
