"""
Ollama LLM implementation.
"""

from typing import Iterator, List, Optional

import httpx

from ..config import LLMConfig
from ..utils.logger import get_logger
from ..utils.text import count_tokens
from .base import LLM

logger = get_logger(__name__)


class OllamaLLM(LLM):
    """Ollama LLM provider for local models."""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize Ollama LLM.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.base_url = config.ollama_base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
            logger.info(f"Connected to Ollama at {self.base_url}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}")
            logger.warning("Make sure Ollama is running: ollama serve")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate text from prompt."""
        # Prepare request
        data = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            }
        }
        
        if stop:
            data["options"]["stop"] = stop
        
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(self.api_url, json=data)
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "")
        
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """Stream generated text."""
        # Prepare request
        data = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature or self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            }
        }
        
        if stop:
            data["options"]["stop"] = stop
        
        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream("POST", self.api_url, json=data) as response:
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if line:
                            try:
                                import json
                                result = json.loads(line)
                                if "response" in result:
                                    yield result["response"]
                            except json.JSONDecodeError:
                                continue
        
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise
    
    def chat(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Chat completion with conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated response
        """
        data = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            }
        }
        
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(self.chat_url, json=data)
                response.raise_for_status()
                
                result = response.json()
                if "message" in result:
                    return result["message"].get("content", "")
                return ""
        
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens (approximate using tiktoken)."""
        return count_tokens(text)
