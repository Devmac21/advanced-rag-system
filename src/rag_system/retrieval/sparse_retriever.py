"""
Sparse retrieval using BM25.
"""

from typing import List

from rank_bm25 import BM25Okapi

from ..config import RetrievalConfig
from ..models import Chunk, RetrievedChunk
from ..utils.logger import get_logger
from .base import Retriever

logger = get_logger(__name__)


class SparseRetriever(Retriever):
    """Sparse retrieval using BM25."""
    
    def __init__(
        self,
        chunks: List[Chunk],
        config: RetrievalConfig,
    ):
        """
        Initialize sparse retriever.
        
        Args:
            chunks: List of all chunks to search
            config: Retrieval configuration
        """
        self.chunks = chunks
        self.config = config
        
        # Tokenize documents for BM25
        tokenized_corpus = [chunk.content.lower().split() for chunk in chunks]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=config.bm25_k1,
            b=config.bm25_b,
        )
        
        logger.info(f"Initialized BM25 with {len(chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Retrieve using BM25."""
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k results
        top_indices = scores.argsort()[-top_k:][::-1]
        
        retrieved = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            
            # Normalize score to 0-1 range (BM25 scores can be large)
            normalized_score = min(score / 10.0, 1.0)
            
            retrieved_chunk = RetrievedChunk(
                chunk=self.chunks[idx],
                score=normalized_score,
                rank=rank,
            )
            retrieved.append(retrieved_chunk)
        
        logger.debug(f"Sparse retrieval found {len(retrieved)} chunks")
        return retrieved
    
    def update_chunks(self, chunks: List[Chunk]):
        """Update the chunk corpus."""
        self.chunks = chunks
        tokenized_corpus = [chunk.content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        logger.info(f"Updated BM25 with {len(chunks)} chunks")
