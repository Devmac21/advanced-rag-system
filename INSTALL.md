# Installation Guide

This guide will help you set up the Advanced RAG System on your machine.

## Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) GPU with CUDA support for faster inference
- (Recommended) Ollama for local LLM inference

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/advanced-rag.git
cd advanced-rag
```

## Step 2: Create Virtual Environment

### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### On Linux/Mac:
```bash
python -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

### Basic Installation:
```bash
pip install -r requirements.txt
```

### Development Installation:
```bash
pip install -r requirements-dev.txt
```

### GPU Support (Optional):
If you have NVIDIA GPU with CUDA:
```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

## Step 4: Download Required Models

### SpaCy Model:
```bash
python -m spacy download en_core_web_sm
```

### NLTK Data:
The system will automatically download required NLTK data on first run.

## Step 5: Install Ollama (Recommended)

For local LLM inference without API costs:

### On Windows:
Download and install from: https://ollama.ai/download/windows

### On Linux:
```bash
curl https://ollama.ai/install.sh | sh
```

### On Mac:
```bash
brew install ollama
```

### Pull a Model:
```bash
ollama pull llama3.1
# or
ollama pull mistral
```

### Start Ollama Server:
```bash
ollama serve
```

## Step 6: Install the Package (Optional)

For system-wide installation:
```bash
pip install -e .
```

This allows you to use `rag` command from anywhere.

## Step 7: Verify Installation

Run the quickstart example:
```bash
python examples/quickstart.py
```

Or test the CLI:
```bash
python cli.py --help
```

## Alternative: Docker Installation

### Using Docker Compose:
```bash
docker-compose up -d
```

The web UI will be available at: http://localhost:8501

### Manual Docker Build:
```bash
docker build -t advanced-rag .
docker run -p 8501:8501 -v ./documents:/app/documents advanced-rag
```

## Configuration

Create a custom config file:

```bash
cp configs/default.yaml configs/my_config.yaml
# Edit my_config.yaml as needed
```

Use custom config:
```bash
python cli.py chat --config-file configs/my_config.yaml
```

## Troubleshooting

### Issue: "No module named 'rag_system'"
**Solution**: Make sure you're in the project root and virtual environment is activated.

### Issue: "Connection refused to Ollama"
**Solution**: Make sure Ollama is running: `ollama serve`

### Issue: "CUDA out of memory"
**Solution**: Use CPU mode by setting `device: "cpu"` in config, or reduce batch size.

### Issue: "ModuleNotFoundError: No module named 'faiss'"
**Solution**: Install FAISS: `pip install faiss-cpu`

### Issue: SpaCy model not found
**Solution**: Download model: `python -m spacy download en_core_web_sm`

## System Requirements

### Minimum:
- CPU: 2+ cores
- RAM: 4GB
- Disk: 5GB free space

### Recommended:
- CPU: 4+ cores
- RAM: 8GB+
- GPU: NVIDIA GPU with 4GB+ VRAM (optional)
- Disk: 10GB+ free space

## Next Steps

After installation:

1. **Quick Test**: Run `python examples/quickstart.py`
2. **Ingest Documents**: `python cli.py ingest ./your_documents`
3. **Start Chatting**: `python cli.py chat`
4. **Web Interface**: `python cli.py serve`

For more information, see the [README.md](README.md) and [documentation](docs/).

## Getting Help

- Check the [FAQ](docs/FAQ.md)
- Open an issue on GitHub
- Join our community discussions

Happy RAGging! 🚀
