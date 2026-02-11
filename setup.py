"""Setup script for advanced-rag-system."""

from pathlib import Path
from setuptools import setup, find_packages

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Advanced RAG System with production-grade features"

setup(
    name="advanced-rag-system",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-grade Retrieval-Augmented Generation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/advanced-rag",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
        "transformers>=4.30.0",
        "faiss-cpu>=1.7.4",
        "chromadb>=0.4.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "pypdf>=3.15.0",
        "python-docx>=1.0.0",
        "markdown>=3.4.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "tiktoken>=0.5.0",
        "rank-bm25>=0.2.2",
        "spacy>=3.6.0",
        "nltk>=3.8.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "rich>=13.0.0",
        "typer>=0.9.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "python-multipart>=0.0.6",
        "httpx>=0.24.0",
        "tenacity>=8.2.0",
        "loguru>=0.7.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "rag=cli:app",
        ],
    },
)
