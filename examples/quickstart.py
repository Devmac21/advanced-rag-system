"""
Quickstart example for the Advanced RAG System.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_system import Config, RAGPipeline


def main():
    """Run quickstart example."""
    
    print("=" * 60)
    print("Advanced RAG System - Quickstart Example")
    print("=" * 60)
    
    # 1. Initialize pipeline
    print("\n1. Initializing RAG pipeline...")
    config = Config()
    pipeline = RAGPipeline(config)
    
    # 2. Create sample documents
    print("\n2. Creating sample documents...")
    sample_dir = Path("examples/sample_docs")
    sample_dir.mkdir(exist_ok=True, parents=True)
    
    # Create sample files
    with open(sample_dir / "python_intro.txt", "w") as f:
        f.write("""
Python Programming Introduction

Python is a high-level, interpreted programming language known for its simplicity and readability.
It was created by Guido van Rossum and first released in 1991.

Key Features:
- Easy to learn and use
- Extensive standard library
- Support for multiple programming paradigms
- Large and active community
- Cross-platform compatibility

Python is widely used in:
- Web development (Django, Flask)
- Data science and machine learning (pandas, scikit-learn, TensorFlow)
- Automation and scripting
- Scientific computing
- Artificial Intelligence
""")
    
    with open(sample_dir / "machine_learning.txt", "w") as f:
        f.write("""
Machine Learning Basics

Machine learning is a subset of artificial intelligence that focuses on building systems
that can learn from and make decisions based on data.

Types of Machine Learning:

1. Supervised Learning
   - Uses labeled training data
   - Examples: Classification, Regression
   - Algorithms: Linear Regression, Decision Trees, Neural Networks

2. Unsupervised Learning
   - Uses unlabeled data
   - Examples: Clustering, Dimensionality Reduction
   - Algorithms: K-Means, PCA, Autoencoders

3. Reinforcement Learning
   - Learns through trial and error
   - Uses rewards and penalties
   - Examples: Game playing, Robotics

Popular ML frameworks: TensorFlow, PyTorch, scikit-learn
""")
    
    # 3. Ingest documents
    print("\n3. Ingesting documents...")
    num_chunks = pipeline.ingest_directory(str(sample_dir))
    print(f"   ✓ Ingested {num_chunks} chunks")
    
    # 4. Get pipeline stats
    print("\n4. Pipeline statistics:")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"   - {key}: {value}")
    
    # 5. Ask questions
    print("\n5. Asking questions...\n")
    
    questions = [
        "What is Python?",
        "What are the types of machine learning?",
        "What is supervised learning?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 60}")
        print(f"Question {i}: {question}")
        print('=' * 60)
        
        response = pipeline.query(question, top_k=3)
        
        print(f"\nAnswer:\n{response.answer}")
        
        print(f"\nSources ({len(response.sources)}):")
        for j, source in enumerate(response.sources, 1):
            src_file = source.chunk.metadata.get('source', 'Unknown')
            src_name = Path(src_file).name if src_file != 'Unknown' else src_file
            print(f"  [{j}] {src_name} (score: {source.score:.3f})")
        
        print(f"\nMetrics:")
        print(f"  - Retrieval time: {response.retrieval_time:.3f}s")
        print(f"  - Generation time: {response.generation_time:.3f}s")
        print(f"  - Total time: {response.total_time:.3f}s")
    
    print("\n" + "=" * 60)
    print("Quickstart complete!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("  1. Try the CLI: python cli.py chat")
    print("  2. Launch web UI: python cli.py serve")
    print("  3. Ingest your own documents: python cli.py ingest <path>")


if __name__ == "__main__":
    main()
