"""
RAG system evaluator.
"""

from typing import Dict, List
import time

from ..models import EvaluationMetrics, QueryResponse
from ..pipeline import RAGPipeline
from ..utils.logger import get_logger
from .metrics import GenerationMetrics, RetrievalMetrics

logger = get_logger(__name__)


class RAGEvaluator:
    """Evaluate RAG system performance."""
    
    def __init__(self, pipeline: RAGPipeline):
        """
        Initialize evaluator.
        
        Args:
            pipeline: RAG pipeline to evaluate
        """
        self.pipeline = pipeline
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        ground_truth: Dict[str, List[str]],  # query -> relevant chunk IDs
        k_values: List[int] = [1, 3, 5, 10],
    ) -> EvaluationMetrics:
        """
        Evaluate retrieval performance.
        
        Args:
            queries: List of test queries
            ground_truth: Dict mapping queries to relevant chunk IDs
            k_values: K values to evaluate
        
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating retrieval on {len(queries)} queries")
        
        all_precision = {k: [] for k in k_values}
        all_recall = {k: [] for k in k_values}
        all_mrr = []
        retrieval_times = []
        
        for query in queries:
            if query not in ground_truth:
                logger.warning(f"No ground truth for query: {query}")
                continue
            
            relevant_ids = ground_truth[query]
            
            # Retrieve
            start_time = time.time()
            retrieved = self.pipeline.retriever.retrieve(query, top_k=max(k_values))
            retrieval_time = time.time() - start_time
            
            retrieval_times.append(retrieval_time)
            
            # Calculate metrics
            for k in k_values:
                precision = self.retrieval_metrics.precision_at_k(retrieved, relevant_ids, k)
                recall = self.retrieval_metrics.recall_at_k(retrieved, relevant_ids, k)
                
                all_precision[k].append(precision)
                all_recall[k].append(recall)
            
            mrr = self.retrieval_metrics.mrr(retrieved, relevant_ids)
            all_mrr.append(mrr)
        
        # Aggregate results
        metrics = EvaluationMetrics()
        
        for k in k_values:
            if all_precision[k]:
                metrics.precision_at_k[k] = sum(all_precision[k]) / len(all_precision[k])
                metrics.recall_at_k[k] = sum(all_recall[k]) / len(all_recall[k])
        
        if all_mrr:
            metrics.mrr = sum(all_mrr) / len(all_mrr)
        
        if retrieval_times:
            metrics.avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
        
        logger.info(f"Retrieval evaluation complete. MRR: {metrics.mrr:.3f}")
        return metrics
    
    def evaluate_generation(
        self,
        queries: List[str],
        expected_answers: Dict[str, str] = None,
    ) -> EvaluationMetrics:
        """
        Evaluate generation quality.
        
        Args:
            queries: List of test queries
            expected_answers: Optional dict of expected answers
        
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating generation on {len(queries)} queries")
        
        all_faithfulness = []
        all_relevance = []
        generation_times = []
        total_times = []
        
        for query in queries:
            # Generate answer
            response = self.pipeline.query(query)
            
            generation_times.append(response.generation_time)
            total_times.append(response.total_time)
            
            # Calculate faithfulness (answer vs context)
            if response.sources:
                context = " ".join([s.chunk.content for s in response.sources])
                faithfulness = self.generation_metrics.faithfulness(response.answer, context)
                all_faithfulness.append(faithfulness)
            
            # Calculate relevance (answer vs query)
            relevance = self.generation_metrics.relevance(response.answer, query)
            all_relevance.append(relevance)
        
        # Aggregate results
        metrics = EvaluationMetrics()
        
        if all_faithfulness:
            metrics.faithfulness = sum(all_faithfulness) / len(all_faithfulness)
        
        if all_relevance:
            metrics.relevance = sum(all_relevance) / len(all_relevance)
        
        if generation_times:
            metrics.avg_generation_time = sum(generation_times) / len(generation_times)
        
        if total_times:
            metrics.avg_total_time = sum(total_times) / len(total_times)
        
        logger.info(f"Generation evaluation complete. Faithfulness: {metrics.faithfulness:.3f}, Relevance: {metrics.relevance:.3f}")
        return metrics
    
    def run_full_evaluation(
        self,
        queries: List[str],
        ground_truth: Dict[str, List[str]] = None,
        k_values: List[int] = [1, 3, 5, 10],
    ) -> EvaluationMetrics:
        """
        Run full evaluation (retrieval + generation).
        
        Args:
            queries: List of test queries
            ground_truth: Optional dict mapping queries to relevant chunk IDs
            k_values: K values to evaluate for retrieval
        
        Returns:
            Combined evaluation metrics
        """
        logger.info("Running full RAG evaluation")
        
        # Evaluate retrieval if ground truth provided
        if ground_truth:
            retrieval_metrics = self.evaluate_retrieval(queries, ground_truth, k_values)
        else:
            retrieval_metrics = EvaluationMetrics()
        
        # Evaluate generation
        generation_metrics = self.evaluate_generation(queries)
        
        # Combine metrics
        combined = EvaluationMetrics(
            precision_at_k=retrieval_metrics.precision_at_k,
            recall_at_k=retrieval_metrics.recall_at_k,
            mrr=retrieval_metrics.mrr,
            ndcg=retrieval_metrics.ndcg,
            faithfulness=generation_metrics.faithfulness,
            relevance=generation_metrics.relevance,
            coherence=generation_metrics.coherence,
            avg_retrieval_time=generation_metrics.avg_retrieval_time,
            avg_generation_time=generation_metrics.avg_generation_time,
            avg_total_time=generation_metrics.avg_total_time,
        )
        
        logger.info("Full evaluation complete")
        return combined
