"""
Advanced RAG System
A production-grade Retrieval-Augmented Generation system with cutting-edge features.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .config import Config
from .pipeline import RAGPipeline
from .models import QueryResponse, Document, Chunk

__all__ = [
    "Config",
    "RAGPipeline",
    "QueryResponse",
    "Document",
    "Chunk",
]
