"""
Factory for creating embedding models.
"""

from ..config import EmbeddingConfig
from ..utils.logger import get_logger
from .base import EmbeddingModel
from .sentence_transformer import SentenceTransformerEmbedding

logger = get_logger(__name__)


class EmbeddingFactory:
    """Factory for creating embedding models."""
    
    @staticmethod
    def create(config: EmbeddingConfig) -> EmbeddingModel:
        """
        Create an embedding model based on configuration.
        
        Args:
            config: Embedding configuration
        
        Returns:
            Embedding model instance
        """
        # For now, we only support Sentence Transformers
        # Can extend to support OpenAI, Cohere, etc.
        logger.info(f"Creating embedding model: {config.model}")
        return SentenceTransformerEmbedding(config)
