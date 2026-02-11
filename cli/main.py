"""LLMQ CLI - Single entry point for LLM Quality Gate."""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm.factory import LLMFactory
from evals.service import EvaluationService

# Initialize Typer app and Rich console
app = typer.Typer(
    name="llmq",
    help="LLMQ - LLM Quality Gate CLI\n\nA provider-agnostic evaluation framework for LLM applications.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def configure_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _base_url(host: str, port: int) -> str:
    """Construct base API URL."""
    return f"http://{host}:{port}/api/v1"


def _handle_api_error(e: Exception, operation: str) -> None:
    """Handle API errors with developer-friendly messages."""
    if isinstance(e, httpx.ConnectError):
        rprint("[red]ERROR: Cannot connect to LLMQ API server.[/red]")
        rprint("[yellow]TIP: Make sure the dashboard is running:[/yellow] [bold]llmq dashboard[/bold]")
        rprint("   [dim]Or check if the server is running on a different port.[/dim]")
    elif isinstance(e, httpx.TimeoutException):
        rprint(f"[red]ERROR: Request timed out while {operation}.[/red]")
        rprint("[yellow]TIP: The evaluation might be taking longer than expected.[/yellow]")
        rprint("   [dim]Try increasing timeout with --timeout or check server logs.[/dim]")
    elif isinstance(e, httpx.HTTPStatusError):
        rprint(f"[red]ERROR: API error ({e.response.status_code}) while {operation}:[/red]")
        try:
            error_detail = e.response.json()
            rprint(f"   {error_detail.get('detail', 'Unknown error')}")
        except:
            rprint(f"   {e.response.text}")
        
        if e.response.status_code == 422:
            rprint("[yellow]TIP: Check your request parameters and try again.[/yellow]")
        elif e.response.status_code == 500:
            rprint("[yellow]TIP: Server error - check the dashboard logs for details.[/yellow]")
    else:
        rprint(f"[red]ERROR: Unexpected error while {operation}:[/red] {e}")
        rprint("[yellow]TIP: Check your network connection and API server status.[/yellow]")
    raise typer.Exit(1)


def _print_evaluation_summary(result: dict):
    """Print evaluation results in a formatted way."""
    payload = result.get("result", {})
    gate = payload.get("quality_gate", {})
    
    # Create a results table
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Provider/Model", f"{result.get('provider')}/{result.get('model')}")
    table.add_row("Quality Gate", "PASSED" if gate.get('passed') else "FAILED")
    table.add_row("Overall Score", f"{gate.get('overall_score', 0):.2%}")
    table.add_row("Run ID", str(result.get('run_id')))
    
    console.print(table)


@app.command()
def eval(
    provider: str = typer.Option(..., "--provider", "-p", help="LLM provider to evaluate (e.g., groq, openai, claude)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific model to use (defaults to provider's default)"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Path to evaluation dataset (defaults to evals/dataset.json)"),
    config: str = typer.Option("config.yaml", "--config", "-c", help="Configuration file path"),
    workers: int = typer.Option(5, "--workers", help="Number of parallel workers"),
    timeout: int = typer.Option(30, "--timeout", help="Request timeout in seconds"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM-as-judge metrics (faster, less accurate)"),
    no_db: bool = typer.Option(False, "--no-db", help="Skip saving results to database"),
    non_deterministic: bool = typer.Option(False, "--non-deterministic", help="Allow non-deterministic evaluation (temperature > 0)"),
    fail_on_gate: bool = typer.Option(False, "--fail-on-gate", help="Exit with error code 1 if quality gate fails (useful for CI)"),
    api_host: str = typer.Option("127.0.0.1", "--api-host", help="API server host"),
    api_port: int = typer.Option(8000, "--api-port", help="API server port"),
):
    """Run LLM evaluation with quality gates."""
    body = {
        "provider": provider,
        "model": model,
        "dataset": dataset,
        "config": config,
        "workers": workers,
        "timeout": timeout,
        "no_judge": no_judge,
        "no_db": no_db,
        "non_deterministic": non_deterministic,
    }
    
    try:
        with console.status(f"[bold green]Evaluating {provider}..."):
            with httpx.Client(timeout=600) as client:
                resp = client.post(f"{_base_url(api_host, api_port)}/evaluate", json=body)
                resp.raise_for_status()
                data = resp.json()
    except Exception as e:
        _handle_api_error(e, "running evaluation")

    if "results" in data:
        for item in data["results"]:
            _print_evaluation_summary(item)
    else:
        _print_evaluation_summary(data)
        if fail_on_gate and not data.get("result", {}).get("quality_gate", {}).get("passed"):
            rprint("\n[red]Quality gate failed - exiting with error code 1[/red]")
            raise typer.Exit(1)


@app.command()
def providers(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Configuration file path"),
    api_host: str = typer.Option("127.0.0.1", "--api-host", help="API server host"),
    api_port: int = typer.Option(8000, "--api-port", help="API server port"),
):
    """List available LLM providers and their status."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{_base_url(api_host, api_port)}/providers", params={"config": config})
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        _handle_api_error(e, "fetching providers")
        
    # Create providers table
    table = Table(title="Available LLM Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Default Model", style="yellow")
    
    for provider in data.get("providers", []):
        status = "Available" if provider.get("available") else "Not configured"
        table.add_row(
            provider['name'],
            status,
            provider.get('default_model', 'N/A')
        )
    
    console.print(table)


@app.command()
def dashboard(
    port: int = typer.Option(8000, "--port", help="Server port"),
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
):
    """Launch the LLMQ web dashboard and API server."""
    rprint(f"[bold green]Starting LLMQ dashboard on http://{host}:{port}[/bold green]")
    rprint(f"[cyan]Dashboard:[/cyan] http://{host}:{port}")
    rprint(f"[cyan]API:[/cyan] http://{host}:{port}/api/v1")
    rprint(f"[cyan]API Docs:[/cyan] http://{host}:{port}/docs")
    rprint()
    
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "dashboard.app:app",
                "--host",
                host,
                "--port",
                str(port),
                "--reload" if reload else "--no-access-log",
            ],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
    except KeyboardInterrupt:
        rprint("\n[yellow]Dashboard stopped[/yellow]")
    except FileNotFoundError:
        rprint("[red]ERROR: uvicorn not found.[/red] Install with: [bold]pip install uvicorn[/bold]")
        raise typer.Exit(1)


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
):
    """Initialize a new LLMQ project with configuration files."""
    import shutil
    import json
    
    # Define files to create
    config_file = Path("llmq.yaml")
    dataset_file = Path("evals/dataset.json")
    env_example_file = Path(".env.example")
    
    files_to_create = [
        (config_file, "configuration"),
        (dataset_file, "dataset"),
        (env_example_file, "environment example")
    ]
    
    # Check if files exist
    existing_files = []
    for file_path, description in files_to_create:
        if file_path.exists():
            existing_files.append((file_path, description))
    
    if existing_files and not force:
        rprint("[red]ERROR: The following files already exist:[/red]")
        for file_path, description in existing_files:
            rprint(f"  • {file_path} ({description})")
        rprint("[yellow]TIP: Use --force to overwrite existing files.[/yellow]")
        raise typer.Exit(1)
    
    try:
        # Create llmq.yaml
        llmq_config = """# LLMQ Configuration
llm:
  default_provider: "groq"
  temperature: 0.0
  max_tokens: 1000

providers:
  groq:
    api_key_env: "GROQ_API_KEY"
    model: "llama-3.1-8b-instant"
  
  openai:
    api_key_env: "OPENAI_API_KEY"
    model: "gpt-3.5-turbo"
    
  claude:
    api_key_env: "ANTHROPIC_API_KEY"
    model: "claude-3-haiku-20240307"
    
  gemini:
    api_key_env: "GOOGLE_API_KEY"
    model: "gemini-1.5-flash"
    
  huggingface:
    api_key_env: "HUGGINGFACE_API_KEY"
    model: "microsoft/DialoGPT-medium"
    
  openrouter:
    api_key_env: "OPENROUTER_API_KEY"
    model: "meta-llama/llama-3.1-8b-instruct:free"

quality_gates:
  task_success_threshold: 0.8
  relevance_threshold: 0.7
  hallucination_threshold: 0.1
"""
        config_file.write_text(llmq_config)
        rprint(f"[green]SUCCESS: Created {config_file}[/green]")
        
        # Create evals directory and dataset.json
        dataset_file.parent.mkdir(exist_ok=True)
        
        minimal_dataset = {
            "test_cases": [
                {
                    "id": "example_1",
                    "task_type": "question_answering",
                    "input": "What is the capital of France?",
                    "expected_output": "Paris",
                    "context": "Geography question about European capitals",
                    "reference": "Paris is the capital and largest city of France."
                },
                {
                    "id": "example_2", 
                    "task_type": "summarization",
                    "input": "Summarize this text: The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is commonly used for typing practice.",
                    "expected_output": "A sentence containing all alphabet letters used for typing practice.",
                    "context": "Text summarization task",
                    "reference": "The sentence is a pangram used for typing practice."
                },
                {
                    "id": "example_3",
                    "task_type": "classification", 
                    "input": "Classify the sentiment: I love this product! It works perfectly.",
                    "expected_output": "positive",
                    "context": "Sentiment analysis task",
                    "reference": "The text expresses satisfaction and positive feelings."
                },
                {
                    "id": "example_4",
                    "task_type": "reasoning",
                    "input": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                    "expected_output": "5 minutes",
                    "context": "Logic reasoning problem",
                    "reference": "Each machine makes 1 widget in 5 minutes, so 100 machines make 100 widgets in 5 minutes."
                },
                {
                    "id": "example_5",
                    "task_type": "translation",
                    "input": "Translate to Spanish: Hello, how are you?",
                    "expected_output": "Hola, ¿cómo estás?",
                    "context": "English to Spanish translation",
                    "reference": "Basic greeting translation from English to Spanish."
                }
            ]
        }
        
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(minimal_dataset, f, indent=2, ensure_ascii=False)
        rprint(f"[green]SUCCESS: Created {dataset_file} with 5 test cases[/green]")
        
        # Create .env.example
        env_example_content = """# LLMQ Environment Variables
# Copy this file to .env and fill in your API keys

# Groq (Free tier available - recommended for getting started)
GROQ_API_KEY=your_groq_api_key_here

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google Gemini
GOOGLE_API_KEY=your_google_api_key_here

# HuggingFace
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# OpenRouter (Access to multiple models)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Local providers (no API key needed)
# Ollama: Install from https://ollama.ai
# LocalAI: Install from https://localai.io
"""
        env_example_file.write_text(env_example_content)
        rprint(f"[green]SUCCESS: Created {env_example_file}[/green]")
        
        rprint("\n[bold green]LLMQ project initialized successfully![/bold green]")
        rprint("\n[cyan]Next steps:[/cyan]")
        rprint("1. Copy [bold].env.example[/bold] to [bold].env[/bold] and add your API keys")
        rprint("2. Review and customize [bold]llmq.yaml[/bold] configuration")
        rprint("3. Run [bold]llmq doctor[/bold] to verify your setup")
        rprint("4. Run [bold]llmq eval --provider groq[/bold] to test with the sample dataset")
        
    except Exception as e:
        rprint(f"[red]ERROR: Failed to initialize project:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def doctor():
    """Run diagnostic checks on LLMQ installation and configuration."""
    import os
    import json
    from rich.table import Table
    
    rprint("[bold cyan]LLMQ Doctor - System Health Check[/bold cyan]\n")
    
    critical_issues = 0
    warnings = 0
    
    # Create status table
    table = Table(title="System Status", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="white", width=20)
    table.add_column("Status", width=15)
    table.add_column("Details", style="dim")
    
    # Check 1: Configuration file
    config_files = ["llmq.yaml", "config.yaml"]  # Support both names
    config_found = None
    for config_file in config_files:
        if Path(config_file).exists():
            config_found = config_file
            break
    
    if config_found:
        table.add_row("Configuration", "[green]OK[/green]", f"Found {config_found}")
    else:
        table.add_row("Configuration", "[red]ERROR[/red]", "No config file found")
        critical_issues += 1
    
    # Check 2: Dataset
    dataset_files = ["evals/dataset.json", "dataset.json"]
    dataset_found = None
    test_case_count = 0
    
    for dataset_file in dataset_files:
        dataset_path = Path(dataset_file)
        if dataset_path.exists():
            dataset_found = dataset_file
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                    test_case_count = len(dataset.get('test_cases', []))
            except Exception as e:
                test_case_count = 0
            break
    
    if dataset_found and test_case_count > 0:
        table.add_row("Dataset", "[green]OK[/green]", f"{test_case_count} test cases found")
    elif dataset_found:
        table.add_row("Dataset", "[yellow]WARNING[/yellow]", "Dataset file empty or invalid")
        warnings += 1
    else:
        table.add_row("Dataset", "[red]ERROR[/red]", "No dataset found")
        critical_issues += 1
    
    # Check 3: API Keys and Providers
    api_keys_status = {
        "Groq": os.getenv("GROQ_API_KEY"),
        "OpenAI": os.getenv("OPENAI_API_KEY"), 
        "Claude": os.getenv("ANTHROPIC_API_KEY"),
        "Gemini": os.getenv("GOOGLE_API_KEY"),
        "HuggingFace": os.getenv("HUGGINGFACE_API_KEY"),
        "OpenRouter": os.getenv("OPENROUTER_API_KEY")
    }
    
    configured_providers = []
    missing_providers = []
    
    for provider, api_key in api_keys_status.items():
        if api_key and api_key.strip():
            configured_providers.append(provider)
            status = "[green]Configured[/green]"
            details = "API key found"
        else:
            missing_providers.append(provider)
            status = "[red]Missing API key[/red]"
            details = "Set in .env file"
        
        table.add_row(provider, status, details)
    
    # Always available local providers
    table.add_row("Ollama", "[blue]Local[/blue]", "No API key needed")
    table.add_row("LocalAI", "[blue]Local[/blue]", "No API key needed")
    
    # Check 4: Embedding model
    embedding_model = "all-MiniLM-L6-v2"  # Default model used by LLMQ
    try:
        # Try to import sentence-transformers to check if embeddings are available
        import sentence_transformers
        table.add_row("Embeddings", "[green]OK[/green]", f"{embedding_model}")
    except ImportError:
        table.add_row("Embeddings", "[yellow]WARNING[/yellow]", "sentence-transformers not installed")
        warnings += 1
    
    # Check 5: Python environment
    if sys.version_info >= (3, 8):
        table.add_row("Python", "[green]OK[/green]", f"v{sys.version.split()[0]}")
    else:
        table.add_row("Python", "[red]ERROR[/red]", f"v{sys.version.split()[0]} (requires >=3.8)")
        critical_issues += 1
    
    # Check 6: Required packages
    required_packages = ["typer", "httpx", "pydantic", "fastapi", "uvicorn"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if not missing_packages:
        table.add_row("Dependencies", "[green]OK[/green]", "All packages installed")
    else:
        table.add_row("Dependencies", "[red]ERROR[/red]", f"Missing: {', '.join(missing_packages)}")
        critical_issues += 1
    
    # Print the table
    console.print(table)
    
    # Summary
    rprint(f"\n[bold cyan]Summary:[/bold cyan]")
    
    if critical_issues == 0 and warnings == 0:
        rprint("[bold green]All systems operational! LLMQ is ready to use.[/bold green]")
        if configured_providers:
            rprint(f"[green]Configured providers:[/green] {', '.join(configured_providers)}")
        rprint("\n[cyan]Quick start:[/cyan]")
        if configured_providers:
            provider = configured_providers[0].lower()
            rprint(f"   • Run [bold]llmq eval --provider {provider}[/bold] to test evaluation")
        rprint("   • Run [bold]llmq dashboard[/bold] to start the web interface")
        
    elif critical_issues == 0:
        rprint(f"[yellow]System ready with {warnings} warning(s).[/yellow]")
        if configured_providers:
            rprint(f"[green]Configured providers:[/green] {', '.join(configured_providers)}")
        else:
            rprint("[yellow]No API keys configured. Add keys to .env file or use local providers.[/yellow]")
            
    else:
        rprint(f"[red]Found {critical_issues} critical issue(s) and {warnings} warning(s).[/red]")
        rprint("\n[cyan]Suggested fixes:[/cyan]")
        
        if not config_found:
            rprint("   • Run [bold]llmq init[/bold] to create configuration files")
        if not dataset_found:
            rprint("   • Run [bold]llmq init[/bold] to create sample dataset")
        if missing_packages:
            rprint("   • Run [bold]pip install -e .[/bold] to install missing packages")
        if not configured_providers:
            rprint("   • Copy [bold].env.example[/bold] to [bold].env[/bold] and add API keys")
        
        # Exit with error code for CI
        raise typer.Exit(1)


@app.command()
def runs(
    provider: Optional[str] = typer.Option(None, "--provider", help="Filter by provider name"),
    model: Optional[str] = typer.Option(None, "--model", help="Filter by model name"),
    limit: int = typer.Option(50, "--limit", help="Maximum number of runs to return"),
    offset: int = typer.Option(0, "--offset", help="Number of runs to skip"),
    api_host: str = typer.Option("127.0.0.1", "--api-host", help="API server host"),
    api_port: int = typer.Option(8000, "--api-port", help="API server port"),
):
    """List historical evaluation runs."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{_base_url(api_host, api_port)}/runs",
                params={"provider": provider, "model": model, "limit": limit, "offset": offset},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        _handle_api_error(e, "fetching runs")
        
    console.print_json(json.dumps(data, indent=2, default=str))


@app.command()
def compare(
    limit: int = typer.Option(100, "--limit", help="Maximum number of runs to include in comparison"),
    api_host: str = typer.Option("127.0.0.1", "--api-host", help="API server host"),
    api_port: int = typer.Option(8000, "--api-port", help="API server port"),
):
    """Compare performance across different LLM providers."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{_base_url(api_host, api_port)}/compare", params={"limit": limit})
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        _handle_api_error(e, "fetching comparison data")

    # Create comparison table
    table = Table(title="Provider Performance Comparison")
    table.add_column("Provider/Model", style="cyan")
    table.add_column("Runs", justify="right", style="yellow")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Task", justify="right", style="blue")
    table.add_column("Relevance", justify="right", style="blue")
    table.add_column("Hallucination", justify="right", style="red")
    table.add_column("Gate %", justify="right", style="green")
    
    for item in data.get("comparisons", []):
        table.add_row(
            item['provider_model'],
            str(item['runs']),
            f"{item['overall_score']:.1%}",
            f"{item['task_success_score']:.1%}",
            f"{item['relevance_score']:.1%}",
            f"{item['hallucination_score']:.1%}",
            f"{item['quality_gate_pass_rate']:.0%}"
        )
    
    console.print(table)


@app.command()
def settings(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Configuration file path"),
    set_json: Optional[str] = typer.Option(None, "--set", help='Update settings with JSON'),
    api_host: str = typer.Option("127.0.0.1", "--api-host", help="API server host"),
    api_port: int = typer.Option(8000, "--api-port", help="API server port"),
):
    """Get or update quality gate settings."""
    try:
        with httpx.Client(timeout=30) as client:
            if set_json:
                payload = json.loads(set_json)
                resp = client.post(
                    f"{_base_url(api_host, api_port)}/settings",
                    params={"config": config},
                    json=payload,
                )
            else:
                resp = client.get(f"{_base_url(api_host, api_port)}/settings", params={"config": config})
            resp.raise_for_status()
            console.print_json(json.dumps(resp.json(), indent=2))
    except json.JSONDecodeError as e:
        rprint(f"[red]ERROR: Invalid JSON in --set parameter:[/red] {e}")
        example_json = '{"quality_gates": {"task_success_threshold": 0.9}}'
        rprint(f"[yellow]Example:[/yellow] --set '{example_json}'")
        raise typer.Exit(1)
    except Exception as e:
        _handle_api_error(e, "updating settings")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """LLMQ - LLM Quality Gate CLI
    
    A provider-agnostic evaluation framework for LLM applications.
    
    Examples:
      llmq dashboard                          # Start web dashboard
      llmq providers                          # List available providers  
      llmq eval --provider groq               # Run evaluation with Groq
      llmq init                               # Initialize new project
      llmq doctor                             # Run diagnostic checks
    """
    configure_logging(verbose)


if __name__ == "__main__":
    app()
