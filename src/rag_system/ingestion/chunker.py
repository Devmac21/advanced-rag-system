"""
Text chunking strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import nltk
from nltk.tokenize import sent_tokenize

from ..config import ChunkingConfig
from ..models import Chunk, ChunkingStrategy, Document
from ..utils.logger import get_logger
from ..utils.text import count_tokens, split_text

logger = get_logger(__name__)

# Download NLTK data if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


class TextChunker(ABC):
    """Base class for text chunkers."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize chunker with configuration."""
        self.config = config
    
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document into chunks."""
        pass


class FixedSizeChunker(TextChunker):
    """Simple fixed-size chunking with overlap."""
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document into fixed-size chunks."""
        text = document.content
        
        chunks = split_text(
            text,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk = Chunk(
                content=chunk_text,
                document_id=document.id,
                chunk_index=i,
                metadata={
                    **document.metadata,
                    'chunk_index': i,
                    'chunk_method': 'fixed',
                }
            )
            result.append(chunk)
        
        logger.debug(f"Created {len(result)} fixed-size chunks from document {document.id}")
        return result


class RecursiveChunker(TextChunker):
    """Recursive chunking that tries to split at natural boundaries."""
    
    SEPARATORS = [
        "\n\n\n",  # Multiple newlines
        "\n\n",    # Paragraph breaks
        "\n",      # Line breaks
        ". ",      # Sentences
        "! ",
        "? ",
        "; ",
        ", ",      # Clauses
        " ",       # Words
        "",        # Characters
    ]
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document recursively at natural boundaries."""
        chunks = self._split_recursive(document.content, self.config.chunk_size)
        
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk = Chunk(
                content=chunk_text,
                document_id=document.id,
                chunk_index=i,
                metadata={
                    **document.metadata,
                    'chunk_index': i,
                    'chunk_method': 'recursive',
                }
            )
            result.append(chunk)
        
        logger.debug(f"Created {len(result)} recursive chunks from document {document.id}")
        return result
    
    def _split_recursive(
        self,
        text: str,
        chunk_size: int,
        separators: Optional[List[str]] = None,
    ) -> List[str]:
        """Recursively split text."""
        if separators is None:
            separators = self.SEPARATORS.copy()
        
        if not text:
            return []
        
        # Check if text is small enough
        if len(text) <= chunk_size:
            return [text]
        
        # Try each separator
        for i, separator in enumerate(separators):
            if separator == "":
                # Last resort: character-level split
                return split_text(text, chunk_size, self.config.chunk_overlap)
            
            if separator in text:
                # Split by this separator
                splits = text.split(separator)
                
                # Rebuild with separator
                chunks = []
                current_chunk = ""
                
                for split in splits:
                    test_chunk = current_chunk + split + separator
                    
                    if len(test_chunk) <= chunk_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        
                        # If split itself is too large, recurse with next separator
                        if len(split) > chunk_size:
                            sub_chunks = self._split_recursive(
                                split,
                                chunk_size,
                                separators[i + 1:],
                            )
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = split + separator
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                return chunks
        
        # Fallback
        return split_text(text, chunk_size, self.config.chunk_overlap)


class SemanticChunker(TextChunker):
    """Semantic chunking based on sentence similarity."""
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document based on semantic similarity."""
        # Split into sentences
        sentences = sent_tokenize(document.content)
        
        if not sentences:
            return []
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_size + sentence_size > self.config.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Create Chunk objects
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk = Chunk(
                content=chunk_text,
                document_id=document.id,
                chunk_index=i,
                metadata={
                    **document.metadata,
                    'chunk_index': i,
                    'chunk_method': 'semantic',
                }
            )
            result.append(chunk)
        
        logger.debug(f"Created {len(result)} semantic chunks from document {document.id}")
        return result


class ParentChildChunker(TextChunker):
    """Parent-child chunking for hierarchical retrieval."""
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Create parent and child chunks."""
        # First create parent chunks (larger)
        parent_chunks = split_text(
            document.content,
            chunk_size=self.config.parent_chunk_size,
            chunk_overlap=100,
        )
        
        all_chunks = []
        
        for parent_idx, parent_text in enumerate(parent_chunks):
            # Create parent chunk
            parent_chunk = Chunk(
                content=parent_text,
                document_id=document.id,
                chunk_index=parent_idx,
                metadata={
                    **document.metadata,
                    'chunk_index': parent_idx,
                    'chunk_type': 'parent',
                    'chunk_method': 'parent_child',
                }
            )
            all_chunks.append(parent_chunk)
            
            # Create child chunks from parent
            child_texts = split_text(
                parent_text,
                chunk_size=self.config.child_chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            
            for child_idx, child_text in enumerate(child_texts):
                child_chunk = Chunk(
                    content=child_text,
                    document_id=document.id,
                    parent_chunk_id=parent_chunk.id,
                    chunk_index=child_idx,
                    metadata={
                        **document.metadata,
                        'parent_chunk_id': parent_chunk.id,
                        'chunk_index': child_idx,
                        'chunk_type': 'child',
                        'chunk_method': 'parent_child',
                    }
                )
                all_chunks.append(child_chunk)
        
        logger.debug(f"Created {len(all_chunks)} parent-child chunks from document {document.id}")
        return all_chunks


class ChunkerFactory:
    """Factory for creating chunkers based on strategy."""
    
    @staticmethod
    def create_chunker(config: ChunkingConfig) -> TextChunker:
        """Create chunker based on configuration."""
        strategy = config.strategy
        
        if strategy == ChunkingStrategy.FIXED:
            return FixedSizeChunker(config)
        elif strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveChunker(config)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return SemanticChunker(config)
        elif strategy == ChunkingStrategy.PARENT_CHILD:
            return ParentChildChunker(config)
        else:
            logger.warning(f"Unknown chunking strategy: {strategy}, using recursive")
            return RecursiveChunker(config)
