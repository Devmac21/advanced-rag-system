"""
Basic tests for the RAG pipeline.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_system import Config, RAGPipeline
from rag_system.models import Document, Chunk


class TestRAGPipeline:
    """Test RAG pipeline functionality."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a test pipeline."""
        config = Config()
        config.vector_store.collection_name = "test_collection"
        return RAGPipeline(config)
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes correctly."""
        assert pipeline is not None
        assert pipeline.embedding_model is not None
        assert pipeline.vector_store is not None
        assert pipeline.llm is not None
    
    def test_get_stats(self, pipeline):
        """Test getting pipeline statistics."""
        stats = pipeline.get_stats()
        
        assert "total_chunks" in stats
        assert "embedding_dimension" in stats
        assert "chunking_strategy" in stats
        assert "retrieval_strategy" in stats
        assert stats["total_chunks"] == 0  # Initially empty
    
    def test_document_chunking(self, pipeline):
        """Test document chunking."""
        # Create a test document
        doc = Document(
            content="This is a test document. " * 100,
            source="test.txt"
        )
        
        # Chunk it
        chunks = pipeline.chunker.chunk(doc)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.document_id == doc.id for chunk in chunks)
    
    # Add more tests as needed
    # Note: Full integration tests would require Ollama to be running


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
