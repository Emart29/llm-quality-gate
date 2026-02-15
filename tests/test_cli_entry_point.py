"""Tests for the CLI entry point."""

import subprocess
import sys
from pathlib import Path


def test_llmq_command_available():
    """Test that llmq command is available after installation."""
    result = subprocess.run([sys.executable, "-m", "cli.main", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "LLMQ - LLM Quality Gate CLI" in result.stdout


def test_llmq_doctor_command():
    """Test that llmq doctor command runs with expected status semantics."""
    result = subprocess.run([sys.executable, "-m", "cli.main", "doctor"], capture_output=True, text=True)
    assert result.returncode in (0, 2)
    assert "LLMQ Doctor - System Health Check" in result.stdout


def test_llmq_help_shows_all_commands():
    """Test that help shows all expected commands."""
    result = subprocess.run([sys.executable, "-m", "cli.main", "--help"], capture_output=True, text=True)
    assert result.returncode == 0

    expected_commands = ["eval", "providers", "dashboard", "init", "doctor", "runs", "compare", "settings"]
    for command in expected_commands:
        assert command in result.stdout


def test_llmq_init_help():
    """Test that init command help works."""
    result = subprocess.run([sys.executable, "-m", "cli.main", "init", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Initialize a new LLMQ project" in result.stdout


def test_llmq_eval_missing_config_exit_code_2(tmp_path):
    cli_path = Path(__file__).resolve().parents[1] / "cli" / "main.py"
    result = subprocess.run(
        [sys.executable, str(cli_path), "eval", "--provider", "groq"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "No llmq.yaml found" in result.stdout or "No llmq.yaml found" in result.stderr
