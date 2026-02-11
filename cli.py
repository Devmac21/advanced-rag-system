#!/usr/bin/env python
"""
Command-line interface for the RAG system.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.rag_system import Config, RAGPipeline
from src.rag_system.utils.logger import get_logger

app = typer.Typer(help="Advanced RAG System CLI")
console = Console()
logger = get_logger(__name__)


@app.command()
def ingest(
    source: str = typer.Argument(..., help="File or directory path to ingest"),
    collection: str = typer.Option("default", help="Collection name"),
    config_file: Optional[str] = typer.Option(None, help="Path to config YAML file"),
    recursive: bool = typer.Option(True, help="Recursively search directories"),
):
    """Ingest documents into the RAG system."""
    try:
        # Load config
        if config_file:
            config = Config.from_yaml(config_file)
        else:
            config = Config()
        
        config.vector_store.collection_name = collection
        
        # Initialize pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Initializing RAG pipeline...", total=None)
            pipeline = RAGPipeline(config)
        
        # Ingest
        source_path = Path(source)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if source_path.is_file():
                progress.add_task(f"Ingesting {source}...", total=None)
                num_chunks = pipeline.ingest_file(source)
            else:
                progress.add_task(f"Ingesting directory {source}...", total=None)
                num_chunks = pipeline.ingest_directory(source, recursive=recursive)
        
        console.print(f"\n[green]✓[/green] Successfully ingested {num_chunks} chunks into collection '{collection}'")
        
        # Show stats
        stats = pipeline.get_stats()
        table = Table(title="Pipeline Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in stats.items():
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    collection: str = typer.Option("default", help="Collection name"),
    config_file: Optional[str] = typer.Option(None, help="Path to config YAML file"),
    top_k: int = typer.Option(5, help="Number of chunks to retrieve"),
    stream: bool = typer.Option(False, help="Stream the response"),
):
    """Query the RAG system."""
    try:
        # Load config
        if config_file:
            config = Config.from_yaml(config_file)
        else:
            config = Config()
        
        config.vector_store.collection_name = collection
        
        # Initialize pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Initializing RAG pipeline...", total=None)
            pipeline = RAGPipeline(config)
        
        # Query
        console.print(Panel(f"[bold cyan]Question:[/bold cyan] {question}"))
        
        if stream:
            console.print("\n[bold green]Answer:[/bold green]")
            for chunk in pipeline.stream_query(question, top_k=top_k):
                console.print(chunk, end="")
            console.print("\n")
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Generating answer...", total=None)
                response = pipeline.query(question, top_k=top_k)
            
            # Display answer
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Markdown(response.answer))
            
            # Display sources
            console.print("\n[bold cyan]Sources:[/bold cyan]")
            console.print(response.format_sources())
            
            # Display metrics
            console.print(f"\n[dim]Retrieval: {response.retrieval_time:.2f}s | Generation: {response.generation_time:.2f}s | Total: {response.total_time:.2f}s[/dim]")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@app.command()
def chat(
    collection: str = typer.Option("default", help="Collection name"),
    config_file: Optional[str] = typer.Option(None, help="Path to config YAML file"),
):
    """Start an interactive chat session."""
    try:
        # Load config
        if config_file:
            config = Config.from_yaml(config_file)
        else:
            config = Config()
        
        config.vector_store.collection_name = collection
        
        # Initialize pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Initializing RAG pipeline...", total=None)
            pipeline = RAGPipeline(config)
        
        stats = pipeline.get_stats()
        if stats['total_chunks'] == 0:
            console.print("[yellow]⚠[/yellow] No documents found in collection. Please ingest documents first.")
            raise typer.Exit(1)
        
        # Start chat
        console.print(Panel.fit(
            f"[bold cyan]RAG Chat Interface[/bold cyan]\n"
            f"Collection: {collection} | Chunks: {stats['total_chunks']}\n"
            f"Type 'exit' or 'quit' to end the session",
            border_style="cyan"
        ))
        
        conversation_id = "cli_session"
        
        while True:
            # Get user input
            console.print("\n[bold green]You:[/bold green]", end=" ")
            user_input = input().strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print("[dim]Goodbye![/dim]")
                break
            
            if not user_input:
                continue
            
            # Get response
            console.print("\n[bold cyan]Assistant:[/bold cyan]")
            
            full_answer = ""
            for chunk in pipeline.stream_query(user_input, conversation_id=conversation_id):
                console.print(chunk, end="", highlight=False)
                full_answer += chunk
            
            console.print("\n")
        
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    collection: str = typer.Option("default", help="Collection name"),
    config_file: Optional[str] = typer.Option(None, help="Path to config YAML file"),
):
    """Start the web UI server."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        
        from web.app import create_app
        
        # Load config
        if config_file:
            config = Config.from_yaml(config_file)
        else:
            config = Config()
        
        config.vector_store.collection_name = collection
        
        console.print(f"[green]Starting web UI at http://{host}:{port}[/green]")
        
        app = create_app(config)
        
        import uvicorn
        uvicorn.run(app, host=host, port=port)
        
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@app.command()
def clear(
    collection: str = typer.Option("default", help="Collection name"),
    config_file: Optional[str] = typer.Option(None, help="Path to config YAML file"),
):
    """Clear a collection."""
    try:
        confirm = typer.confirm(f"Are you sure you want to clear collection '{collection}'?")
        
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            return
        
        # Load config
        if config_file:
            config = Config.from_yaml(config_file)
        else:
            config = Config()
        
        config.vector_store.collection_name = collection
        
        # Initialize pipeline
        pipeline = RAGPipeline(config)
        pipeline.clear_collection()
        
        console.print(f"[green]✓[/green] Cleared collection '{collection}'")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
