"""
Groq LLM implementation for fast, free inference.
"""

from typing import Iterator, List, Optional

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from ..config import LLMConfig
from ..utils.logger import get_logger
from ..utils.text import count_tokens
from .base import LLM

logger = get_logger(__name__)


class GroqLLM(LLM):
    """Groq LLM provider for fast inference."""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize Groq LLM.
        
        Args:
            config: LLM configuration
        """
        if not GROQ_AVAILABLE:
            raise ImportError(
                "Groq package not installed. Install with: pip install groq"
            )
        
        self.config = config
        
        # Get API key from config or environment
        api_key = config.groq_api_key or None
        if not api_key:
            import os
            api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable "
                "or provide groq_api_key in config."
            )
        
        self.client = Groq(api_key=api_key)
        
        # Map common model names to Groq models (updated 2026)
        model_mapping = {
            "llama3.1": "llama-3.1-8b-instant",  # Updated to current model
            "llama3": "llama3-8b-8192",
            "mixtral": "mixtral-8x7b-32768",
            "gemma": "gemma2-9b-it",
            "llama3.3": "llama-3.3-70b-versatile",  # Latest model
        }
        
        self.model = model_mapping.get(config.model, config.model)
        logger.info(f"Initialized Groq with model: {self.model}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate text from prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                top_p=self.config.top_p,
                stop=stop,
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise
    
    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """Stream generated text."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                top_p=self.config.top_p,
                stop=stop,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            logger.error(f"Groq streaming failed: {e}")
            raise
    
    def chat(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Chat completion with conversation history."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                top_p=self.config.top_p,
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Groq chat failed: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens (approximate using tiktoken)."""
        return count_tokens(text)
