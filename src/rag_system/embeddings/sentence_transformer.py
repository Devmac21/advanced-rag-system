"""
Sentence Transformer based embeddings.
"""

from typing import List

from sentence_transformers import SentenceTransformer

from ..config import EmbeddingConfig
from ..utils.logger import get_logger
from .base import EmbeddingModel

logger = get_logger(__name__)


class SentenceTransformerEmbedding(EmbeddingModel):
    """Sentence Transformer embedding model."""
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize Sentence Transformer model.
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        
        logger.info(f"Loading embedding model: {config.model}")
        self.model = SentenceTransformer(
            config.model,
            device=config.device,
            cache_folder=config.cache_dir,
        )
        
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Dimension: {self._dimension}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of document texts
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Encode in batches
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
        )
        
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        
        Args:
            text: Query text
        
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
        )
        
        return embedding.tolist()
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
