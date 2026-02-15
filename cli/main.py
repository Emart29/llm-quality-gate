"""LLMQ CLI - Single entry point for LLM Quality Gate."""

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import DEFAULT_CONFIG_NAME, resolve_config_path
from core.errors import ConfigError
from evals.service import EvaluationService

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_CONFIG_ERROR = 2

app = typer.Typer(
    name="llmq",
    help="LLMQ - LLM Quality Gate CLI\n\nA provider-agnostic evaluation framework for LLM applications.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def configure_logging(verbose: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/api/v1"


def _resolve_config_or_exit(config_path: Optional[str]) -> str:
    try:
        return str(resolve_config_path(config_path))
    except ConfigError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_CONFIG_ERROR) from exc


def _print_evaluation_summary(result: dict):
    payload = result.get("result", {})
    gate = payload.get("quality_gate", {})

    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Provider/Model", f"{result.get('provider')}/{result.get('model')}")
    table.add_row("Quality Gate", "PASSED" if gate.get("passed") else "FAILED")
    table.add_row("Overall Score", f"{gate.get('overall_score', 0):.2%}")
    table.add_row("Run ID", str(result.get("run_id")))

    console.print(table)


def _run_eval_local(body: dict) -> dict:
    service = EvaluationService(project_root=Path.cwd())
    return asyncio.run(
        service.evaluate_provider(
            provider=body["provider"],
            model=body.get("model"),
            dataset_path=body.get("dataset"),
            config_path=body["config"],
            workers=body.get("workers", 5),
            timeout=body.get("timeout", 30),
            no_judge=body.get("no_judge", False),
            no_db=body.get("no_db", False),
            non_deterministic=body.get("non_deterministic", False),
        )
    )


def _handle_api_error(e: Exception, operation: str) -> None:
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            detail = e.response.text

        rprint(f"[red]ERROR: API error ({status}) while {operation}.[/red]")
        if detail:
            rprint(f"   {detail}")

        if status == 400 and "No llmq.yaml found" in detail:
            raise typer.Exit(EXIT_CONFIG_ERROR)
        raise typer.Exit(EXIT_FAIL)

    if isinstance(e, httpx.TimeoutException):
        rprint(f"[red]ERROR: Request timed out while {operation}.[/red]")
    else:
        rprint(f"[red]ERROR: Unexpected error while {operation}:[/red] {e}")
    raise typer.Exit(EXIT_FAIL)


@app.command()
def eval(
    provider: str = typer.Option(..., "--provider", "-p", help="LLM provider to evaluate (e.g., groq, openai, claude)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific model to use (defaults to provider's default)"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Path to evaluation dataset (defaults to evals/dataset.json)"),
    config: Optional[str] = typer.Option(None, "--config", "-c", "--config-path", help="Path to llmq.yaml"),
    workers: int = typer.Option(5, "--workers", help="Number of parallel workers"),
    timeout: int = typer.Option(30, "--timeout", help="Request timeout in seconds"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM-as-judge metrics (faster, less accurate)"),
    no_db: bool = typer.Option(False, "--no-db", help="Skip saving results to database"),
    non_deterministic: bool = typer.Option(False, "--non-deterministic", help="Allow non-deterministic evaluation (temperature > 0)"),
    fail_on_gate: bool = typer.Option(True, "--fail-on-gate/--no-fail-on-gate", help="Exit 1 when quality gate fails"),
    api_host: str = typer.Option("127.0.0.1", "--api-host", help="API server host"),
    api_port: int = typer.Option(8000, "--api-port", help="API server port"),
):
    """Run LLM evaluation with quality gates."""
    resolved_config = _resolve_config_or_exit(config)
    body = {
        "provider": provider,
        "model": model,
        "dataset": dataset,
        "config": resolved_config,
        "workers": workers,
        "timeout": timeout,
        "no_judge": no_judge,
        "no_db": no_db,
        "non_deterministic": non_deterministic,
    }

    data = None
    try:
        with console.status(f"[bold green]Evaluating {provider} through API..."):
            with httpx.Client(timeout=600) as client:
                resp = client.post(f"{_base_url(api_host, api_port)}/evaluate", json=body)
                resp.raise_for_status()
                data = resp.json()
    except httpx.ConnectError:
        rprint("[yellow]Dashboard API not reachable. Falling back to local standalone evaluation.[/yellow]")
        try:
            with console.status(f"[bold green]Evaluating {provider} locally..."):
                data = _run_eval_local(body)
        except ConfigError as exc:
            rprint(f"[red]{exc}[/red]")
            raise typer.Exit(EXIT_CONFIG_ERROR) from exc
        except Exception as exc:
            rprint(f"[red]Local evaluation failed:[/red] {exc}")
            raise typer.Exit(EXIT_FAIL) from exc
    except Exception as exc:
        _handle_api_error(exc, "running evaluation")

    if "results" in data:
        for item in data["results"]:
            _print_evaluation_summary(item)
        failed = any(not item.get("result", {}).get("quality_gate", {}).get("passed") for item in data["results"])
    else:
        _print_evaluation_summary(data)
        failed = not data.get("result", {}).get("quality_gate", {}).get("passed")

    if fail_on_gate and failed:
        raise typer.Exit(EXIT_FAIL)


@app.command()
def providers(
    config: Optional[str] = typer.Option(None, "--config", "-c", "--config-path", help="Path to llmq.yaml"),
    api_host: str = typer.Option("127.0.0.1", "--api-host", help="API server host"),
    api_port: int = typer.Option(8000, "--api-port", help="API server port"),
):
    """List available LLM providers and their status."""
    resolved_config = _resolve_config_or_exit(config)
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{_base_url(api_host, api_port)}/providers", params={"config": resolved_config})
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        _handle_api_error(e, "fetching providers")

    table = Table(title="Available LLM Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Default Model", style="yellow")

    for provider_data in data.get("providers", []):
        status = "Available" if provider_data.get("available") else "Not configured"
        table.add_row(provider_data["name"], status, provider_data.get("default_model", "N/A"))

    console.print(table)


@app.command()
def dashboard(
    port: int = typer.Option(8000, "--port", help="Server port"),
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
):
    """Launch the LLMQ web dashboard and API server."""
    rprint(f"[bold green]Starting LLMQ dashboard on http://{host}:{port}[/bold green]")
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
            cwd=str(Path.cwd()),
            check=False,
        )
    except FileNotFoundError:
        rprint("[red]ERROR: uvicorn not found.[/red] Install with: [bold]pip install uvicorn[/bold]")
        raise typer.Exit(EXIT_FAIL)


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
):
    """Initialize a new LLMQ project with configuration files."""
    import shutil

    config_file = Path(DEFAULT_CONFIG_NAME)
    dataset_file = Path("evals/dataset.json")
    env_example_file = Path(".env.example")

    files_to_create = [
        (config_file, "configuration"),
        (dataset_file, "dataset"),
        (env_example_file, "environment example"),
    ]

    existing_files = [(path, desc) for path, desc in files_to_create if path.exists()]
    if existing_files and not force:
        rprint("[red]ERROR: The following files already exist:[/red]")
        for file_path, description in existing_files:
            rprint(f"  • {file_path} ({description})")
        raise typer.Exit(EXIT_FAIL)

    if force:
        for file_path, _ in existing_files:
            if file_path.is_dir():
                shutil.rmtree(file_path)
            else:
                file_path.unlink(missing_ok=True)

    llmq_config = """# LLMQ Configuration
llm:
  default_provider: "groq"
  temperature: 0.0
  max_tokens: 1000

providers:
  groq:
    api_key_env: "GROQ_API_KEY"
    model: "llama-3.1-8b-instant"

quality_gates:
  task_success_threshold: 0.8
  relevance_threshold: 0.7
  hallucination_threshold: 0.1
"""
    config_file.write_text(llmq_config)
    dataset_file.parent.mkdir(exist_ok=True)
    dataset_file.write_text('{"test_cases": []}\n')
    env_example_file.write_text("GROQ_API_KEY=your_groq_api_key_here\n")
    rprint("[green]LLMQ project initialized.[/green]")


@app.command()
def doctor():
    """Run diagnostic checks on LLMQ installation and configuration."""
    rprint("[bold cyan]LLMQ Doctor - System Health Check[/bold cyan]\n")
    try:
        resolved = resolve_config_path()
        rprint(f"[green]Configuration:[/green] {resolved}")
    except ConfigError:
        rprint("[red]No llmq.yaml found. Run `llmq init`.[/red]")
        raise typer.Exit(EXIT_CONFIG_ERROR)


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

    table = Table(title="Provider Performance Comparison")
    table.add_column("Provider/Model", style="cyan")
    table.add_column("Runs", justify="right", style="yellow")
    table.add_column("Score", justify="right", style="green")

    for item in data.get("comparisons", []):
        table.add_row(item["provider_model"], str(item["runs"]), f"{item['overall_score']:.1%}")
    console.print(table)


@app.command()
def settings(
    config: Optional[str] = typer.Option(None, "--config", "-c", "--config-path", help="Path to llmq.yaml"),
    set_json: Optional[str] = typer.Option(None, "--set", help='Update settings with JSON'),
    api_host: str = typer.Option("127.0.0.1", "--api-host", help="API server host"),
    api_port: int = typer.Option(8000, "--api-port", help="API server port"),
):
    """Get or update quality gate settings."""
    resolved_config = _resolve_config_or_exit(config)
    try:
        with httpx.Client(timeout=30) as client:
            if set_json:
                payload = json.loads(set_json)
                resp = client.post(f"{_base_url(api_host, api_port)}/settings", params={"config": resolved_config}, json=payload)
            else:
                resp = client.get(f"{_base_url(api_host, api_port)}/settings", params={"config": resolved_config})
            resp.raise_for_status()
            console.print_json(json.dumps(resp.json(), indent=2))
    except json.JSONDecodeError as e:
        rprint(f"[red]ERROR: Invalid JSON in --set parameter:[/red] {e}")
        raise typer.Exit(EXIT_FAIL)
    except Exception as e:
        _handle_api_error(e, "updating settings")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """LLMQ - LLM Quality Gate CLI."""
    configure_logging(verbose)


if __name__ == "__main__":
    app()
