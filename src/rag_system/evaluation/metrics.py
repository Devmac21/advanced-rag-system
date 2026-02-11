"""
Metrics for evaluating RAG systems.
"""

from typing import List
import numpy as np
from ..models import RetrievedChunk


class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""
    
    @staticmethod
    def precision_at_k(retrieved: List[RetrievedChunk], relevant_ids: List[str], k: int) -> float:
        """
        Calculate precision@k.
        
        Args:
            retrieved: List of retrieved chunks
            relevant_ids: List of IDs of relevant chunks
            k: Number of top results to consider
        
        Returns:
            Precision@k score
        """
        if not retrieved or k == 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_count = sum(1 for chunk in top_k if chunk.chunk.id in relevant_ids)
        
        return relevant_count / k
    
    @staticmethod
    def recall_at_k(retrieved: List[RetrievedChunk], relevant_ids: List[str], k: int) -> float:
        """
        Calculate recall@k.
        
        Args:
            retrieved: List of retrieved chunks
            relevant_ids: List of IDs of relevant chunks
            k: Number of top results to consider
        
        Returns:
            Recall@k score
        """
        if not relevant_ids:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_count = sum(1 for chunk in top_k if chunk.chunk.id in relevant_ids)
        
        return relevant_count / len(relevant_ids)
    
    @staticmethod
    def mrr(retrieved: List[RetrievedChunk], relevant_ids: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            retrieved: List of retrieved chunks
            relevant_ids: List of IDs of relevant chunks
        
        Returns:
            MRR score
        """
        for i, chunk in enumerate(retrieved, 1):
            if chunk.chunk.id in relevant_ids:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved: List[RetrievedChunk], relevance_scores: dict, k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@k.
        
        Args:
            retrieved: List of retrieved chunks
            relevance_scores: Dict mapping chunk IDs to relevance scores
            k: Number of top results to consider
        
        Returns:
            NDCG@k score
        """
        def dcg_at_k(scores: List[float], k: int) -> float:
            scores = np.array(scores[:k])
            if scores.size == 0:
                return 0.0
            discounts = np.log2(np.arange(2, scores.size + 2))
            return np.sum(scores / discounts)
        
        # Get actual scores
        actual_scores = [
            relevance_scores.get(chunk.chunk.id, 0.0)
            for chunk in retrieved[:k]
        ]
        
        # Get ideal scores (sorted)
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        
        # Calculate DCG and IDCG
        actual_dcg = dcg_at_k(actual_scores, k)
        ideal_dcg = dcg_at_k(ideal_scores, k)
        
        if ideal_dcg == 0:
            return 0.0
        
        return actual_dcg / ideal_dcg


class GenerationMetrics:
    """Metrics for evaluating generation quality."""
    
    @staticmethod
    def faithfulness(answer: str, context: str) -> float:
        """
        Estimate faithfulness of answer to context.
        Simple heuristic: ratio of answer words found in context.
        
        Args:
            answer: Generated answer
            context: Source context
        
        Returns:
            Faithfulness score (0-1)
        """
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        if not answer_words:
            return 0.0
        
        overlap = answer_words & context_words
        return len(overlap) / len(answer_words)
    
    @staticmethod
    def relevance(answer: str, query: str) -> float:
        """
        Estimate relevance of answer to query.
        Simple heuristic: word overlap.
        
        Args:
            answer: Generated answer
            query: User query
        
        Returns:
            Relevance score (0-1)
        """
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = query_words & answer_words
        return len(overlap) / len(query_words)
    
    @staticmethod
    def answer_length_score(answer: str, min_length: int = 20, max_length: int = 500) -> float:
        """
        Score answer based on length (penalize too short or too long).
        
        Args:
            answer: Generated answer
            min_length: Minimum desired length
            max_length: Maximum desired length
        
        Returns:
            Length score (0-1)
        """
        length = len(answer)
        
        if length < min_length:
            return length / min_length
        elif length > max_length:
            return max(0, 1 - (length - max_length) / max_length)
        else:
            return 1.0
