"""
Main RAG pipeline that orchestrates all components.
"""

import time
from pathlib import Path
from typing import Iterator, List, Optional

from .config import Config
from .embeddings.factory import EmbeddingFactory
from .generation.factory import LLMFactory
from .ingestion.chunker import ChunkerFactory
from .ingestion.loader import DocumentLoaderFactory
from .models import Chunk, Conversation, Document, QueryResponse, RetrievedChunk
from .retrieval.factory import RetrieverFactory
from .retrieval.query_processor import QueryProcessor
from .retrieval.reranker import Reranker
from .utils.logger import get_logger
from .vector_stores.factory import VectorStoreFactory

logger = get_logger(__name__)


class RAGPipeline:
    """Main RAG pipeline orchestrating all components."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize RAG pipeline.
        
        Args:
            config: Configuration object (uses defaults if not provided)
        """
        self.config = config or Config()
        
        # Initialize components
        logger.info("Initializing RAG pipeline...")
        
        # Embedding model
        self.embedding_model = EmbeddingFactory.create(self.config.embeddings)
        
        # Vector store
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store,
            self.embedding_model.dimension,
        )
        
        # LLM
        self.llm = LLMFactory.create(self.config.llm)
        
        # Query processor
        self.query_processor = QueryProcessor(
            self.config.retrieval,
            self.config.prompts,
        )
        
        # Reranker
        self.reranker = Reranker(self.config.retrieval)
        
        # Document loader
        self.document_loader = DocumentLoaderFactory()
        
        # Chunker
        self.chunker = ChunkerFactory.create_chunker(self.config.chunking)
        
        # Store all chunks for sparse retrieval
        self.all_chunks: List[Chunk] = []
        
        # Retriever (will be initialized when we have chunks)
        self.retriever = None
        
        # Conversation management
        self.conversations: dict[str, Conversation] = {}
        
        logger.info("RAG pipeline initialized successfully")
    
    def ingest_file(self, file_path: str) -> int:
        """
        Ingest a single file.
        
        Args:
            file_path: Path to file
        
        Returns:
            Number of chunks created
        """
        logger.info(f"Ingesting file: {file_path}")
        
        # Load document
        document = self.document_loader.load_file(file_path)
        
        # Chunk document
        chunks = self.chunker.chunk(document)
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_documents(
            [chunk.content for chunk in chunks]
        )
        
        # Add to vector store
        self.vector_store.add_chunks(chunks, embeddings)
        
        # Add to all chunks
        self.all_chunks.extend(chunks)
        
        # Persist
        self.vector_store.persist()
        
        # Reinitialize retriever with new chunks
        self._init_retriever()
        
        logger.info(f"Ingested {len(chunks)} chunks from {file_path}")
        return len(chunks)
    
    def ingest_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_pattern: Optional[str] = None,
    ) -> int:
        """
        Ingest all files from a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to search recursively
            file_pattern: File pattern to match
        
        Returns:
            Total number of chunks created
        """
        logger.info(f"Ingesting directory: {directory}")
        
        # Load all documents
        documents = self.document_loader.load_directory(
            directory,
            recursive=recursive,
            file_pattern=file_pattern,
        )
        
        total_chunks = 0
        
        for document in documents:
            # Chunk document
            chunks = self.chunker.chunk(document)
            
            # Generate embeddings
            embeddings = self.embedding_model.embed_documents(
                [chunk.content for chunk in chunks]
            )
            
            # Add to vector store
            self.vector_store.add_chunks(chunks, embeddings)
            
            # Add to all chunks
            self.all_chunks.extend(chunks)
            
            total_chunks += len(chunks)
        
        # Persist
        self.vector_store.persist()
        
        # Reinitialize retriever with new chunks
        self._init_retriever()
        
        logger.info(f"Ingested {total_chunks} chunks from {len(documents)} documents")
        return total_chunks
    
    def _init_retriever(self):
        """Initialize or reinitialize the retriever."""
        if self.all_chunks:
            self.retriever = RetrieverFactory.create(
                self.config.retrieval,
                self.vector_store,
                self.embedding_model,
                self.all_chunks,
            )
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        conversation_id: Optional[str] = None,
    ) -> QueryResponse:
        """
        Query the RAG system.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            conversation_id: Conversation ID for multi-turn chat
        
        Returns:
            Query response with answer and sources
        """
        if not self.retriever:
            raise ValueError("No documents ingested. Please ingest documents first.")
        
        start_time = time.time()
        
        # Get conversation context if provided
        conversation_context = []
        if conversation_id and conversation_id in self.conversations:
            conversation = self.conversations[conversation_id]
            conversation_context = conversation.get_context(
                self.config.max_conversation_history
            )
        
        # Retrieve relevant chunks
        retrieval_start = time.time()
        
        top_k = top_k or self.config.retrieval.top_k
        
        # Use rerank_top_k if reranking is enabled
        if self.config.retrieval.enable_reranking:
            retrieve_k = self.config.retrieval.rerank_top_k
        else:
            retrieve_k = top_k
        
        retrieved_chunks = self.retriever.retrieve(query, top_k=retrieve_k)
        
        # Re-rank if enabled
        if self.config.retrieval.enable_reranking:
            retrieved_chunks = self.reranker.rerank(query, retrieved_chunks, top_k=top_k)
        
        retrieval_time = time.time() - retrieval_start
        
        # Generate answer
        generation_start = time.time()
        
        answer = self._generate_answer(
            query,
            retrieved_chunks,
            conversation_context,
        )
        
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        # Create response
        response = QueryResponse(
            query=query,
            answer=answer,
            sources=retrieved_chunks,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
        )
        
        # Update conversation if provided
        if conversation_id:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = Conversation()
            
            conversation = self.conversations[conversation_id]
            conversation.add_message("user", query)
            conversation.add_message("assistant", answer)
        
        logger.info(f"Query completed in {total_time:.2f}s (retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)")
        
        return response
    
    def stream_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        conversation_id: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Query with streaming response.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            conversation_id: Conversation ID for multi-turn chat
        
        Yields:
            Answer chunks
        """
        if not self.retriever:
            raise ValueError("No documents ingested. Please ingest documents first.")
        
        # Get conversation context if provided
        conversation_context = []
        if conversation_id and conversation_id in self.conversations:
            conversation = self.conversations[conversation_id]
            conversation_context = conversation.get_context(
                self.config.max_conversation_history
            )
        
        # Retrieve relevant chunks
        top_k = top_k or self.config.retrieval.top_k
        
        if self.config.retrieval.enable_reranking:
            retrieve_k = self.config.retrieval.rerank_top_k
        else:
            retrieve_k = top_k
        
        retrieved_chunks = self.retriever.retrieve(query, top_k=retrieve_k)
        
        # Re-rank if enabled
        if self.config.retrieval.enable_reranking:
            retrieved_chunks = self.reranker.rerank(query, retrieved_chunks, top_k=top_k)
        
        # Build prompt
        prompt = self._build_prompt(query, retrieved_chunks, conversation_context)
        
        # Stream answer
        full_answer = ""
        for chunk in self.llm.stream(prompt):
            full_answer += chunk
            yield chunk
        
        # Update conversation if provided
        if conversation_id:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = Conversation()
            
            conversation = self.conversations[conversation_id]
            conversation.add_message("user", query)
            conversation.add_message("assistant", full_answer)
    
    def _generate_answer(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        conversation_context: List[dict],
    ) -> str:
        """Generate answer from retrieved chunks."""
        prompt = self._build_prompt(query, retrieved_chunks, conversation_context)
        answer = self.llm.generate(prompt)
        return answer
    
    def _build_prompt(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        conversation_context: List[dict],
    ) -> str:
        """Build prompt from query and retrieved chunks."""
        # Build context from retrieved chunks
        context_parts = []
        for i, retrieved in enumerate(retrieved_chunks, 1):
            source = retrieved.chunk.metadata.get('source', 'Unknown')
            context_parts.append(f"[{i}] {retrieved.chunk.content}\n(Source: {source})")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        if conversation_context:
            # Multi-turn conversation
            prompt_parts = [self.config.llm.system_prompt]
            
            for msg in conversation_context:
                role = msg["role"]
                content = msg["content"]
                prompt_parts.append(f"{role.capitalize()}: {content}")
            
            # Add context and current query
            prompt_parts.append(f"\nContext:\n{context}\n")
            prompt_parts.append(f"User: {query}")
            prompt_parts.append("Assistant:")
            
            prompt = "\n\n".join(prompt_parts)
        else:
            # Single query
            prompt = self.config.prompts.qa_template.format(
                context=context,
                query=query,
            )
        
        return prompt
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        self.vector_store.delete_collection()
        self.all_chunks = []
        self.retriever = None
        logger.info("Cleared collection")
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "total_chunks": len(self.all_chunks),
            "vector_store_chunks": self.vector_store.get_chunk_count(),
            "embedding_dimension": self.embedding_model.dimension,
            "chunking_strategy": self.config.chunking.strategy,
            "retrieval_strategy": self.config.retrieval.strategy,
            "llm_provider": self.config.llm.provider,
            "llm_model": self.config.llm.model,
        }
