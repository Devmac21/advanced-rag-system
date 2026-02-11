# Contributing to Advanced RAG System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/advanced-rag.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
```

## Code Style

We follow PEP 8 and use the following tools:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run before committing:

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

## Pull Request Guidelines

- Write clear, descriptive commit messages
- Include tests for new features
- Update documentation as needed
- Keep PRs focused on a single feature/fix
- Ensure all tests pass
- Add yourself to CONTRIBUTORS.md

## Testing

- Write unit tests for new functionality
- Ensure test coverage remains high
- Test edge cases and error conditions

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/rag_system --cov-report=html
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all functions/classes
- Update API documentation if needed
- Include examples for new features

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces
- Minimal code example

## Feature Requests

We welcome feature requests! Please:

- Check existing issues first
- Provide clear use case
- Explain expected behavior
- Consider implementation details

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on the issue, not the person

## Questions?

Feel free to open an issue for questions or join our discussions!

Thank you for contributing! 🎉
