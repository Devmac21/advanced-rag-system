"""
Base retriever interface.
"""

from abc import ABC, abstractmethod
from typing import List

from ..models import Chunk, RetrievedChunk


class Retriever(ABC):
    """Base class for retrievers."""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
        
        Returns:
            List of retrieved chunks with scores
        """
        pass
