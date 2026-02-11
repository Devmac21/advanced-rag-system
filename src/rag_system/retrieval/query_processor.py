"""
Query processing utilities (expansion, HyDE, etc.).
"""

from typing import List

from ..config import PromptConfig, RetrievalConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class QueryProcessor:
    """Process and enhance queries for better retrieval."""
    
    def __init__(
        self,
        retrieval_config: RetrievalConfig,
        prompt_config: PromptConfig,
    ):
        """
        Initialize query processor.
        
        Args:
            retrieval_config: Retrieval configuration
            prompt_config: Prompt configuration
        """
        self.retrieval_config = retrieval_config
        self.prompt_config = prompt_config
    
    def expand_query(self, query: str, llm=None) -> List[str]:
        """
        Expand query into multiple variations.
        
        Args:
            query: Original query
            llm: LLM instance for generating expansions
        
        Returns:
            List of query variations (including original)
        """
        if not self.retrieval_config.enable_query_expansion:
            return [query]
        
        queries = [query]  # Start with original
        
        if llm:
            try:
                # Generate query variations using LLM
                prompt = self.prompt_config.query_expansion_template.format(
                    query=query,
                    num_queries=self.retrieval_config.num_expanded_queries,
                )
                
                response = llm.generate(prompt, max_tokens=200)
                
                # Parse response (assuming one query per line)
                lines = response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    # Remove numbering, bullets, etc.
                    line = line.lstrip('0123456789.-• ')
                    if line and line not in queries:
                        queries.append(line)
                
                logger.debug(f"Expanded query to {len(queries)} variations")
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")
        else:
            # Simple heuristic-based expansion without LLM
            queries.extend(self._simple_expansion(query))
        
        return queries[:self.retrieval_config.num_expanded_queries + 1]
    
    def _simple_expansion(self, query: str) -> List[str]:
        """Simple query expansion without LLM."""
        expansions = []
        
        # Add question variations
        if not any(query.lower().startswith(q) for q in ['what', 'how', 'why', 'when', 'where', 'who']):
            expansions.append(f"What is {query}?")
            expansions.append(f"How does {query} work?")
        
        # Add "explain" variation
        if "explain" not in query.lower():
            expansions.append(f"Explain {query}")
        
        return expansions
    
    def generate_hypothetical_document(self, query: str, llm) -> str:
        """
        Generate hypothetical document (HyDE).
        
        Args:
            query: User query
            llm: LLM instance
        
        Returns:
            Hypothetical document text
        """
        if not self.retrieval_config.enable_hyde or not llm:
            return query
        
        try:
            prompt = self.prompt_config.hyde_template.format(query=query)
            response = llm.generate(prompt, max_tokens=300)
            
            logger.debug("Generated hypothetical document for HyDE")
            return response.strip()
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return query
