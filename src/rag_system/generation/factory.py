"""
Factory for creating LLM instances.
"""

from ..config import LLMConfig
from ..models import LLMProvider
from ..utils.logger import get_logger
from .base import LLM
from .ollama_llm import OllamaLLM

logger = get_logger(__name__)


class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create(config: LLMConfig) -> LLM:
        """
        Create an LLM instance based on configuration.
        
        Args:
            config: LLM configuration
        
        Returns:
            LLM instance
        """
        provider = config.provider
        
        logger.info(f"Creating LLM with provider: {provider}")
        
        if provider == LLMProvider.OLLAMA:
            return OllamaLLM(config)
        elif provider == "groq":
            # Import here to avoid dependency if not using Groq
            from .groq_llm import GroqLLM
            return GroqLLM(config)
        elif provider == LLMProvider.OPENAI:
            # Future implementation
            raise NotImplementedError("OpenAI support coming soon")
        elif provider == LLMProvider.ANTHROPIC:
            # Future implementation
            raise NotImplementedError("Anthropic support coming soon")
        elif provider == LLMProvider.HUGGINGFACE:
            # Future implementation
            raise NotImplementedError("HuggingFace support coming soon")
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
