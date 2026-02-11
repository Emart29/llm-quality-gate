"""Tests for the CLI entry point."""

import subprocess
import sys
from pathlib import Path

def test_llmq_command_available():
    """Test that llmq command is available after installation."""
    result = subprocess.run([sys.executable, "-m", "cli.main", "--help"], 
                          capture_output=True, text=True)
    assert result.returncode == 0
    assert "LLMQ - LLM Quality Gate CLI" in result.stdout

def test_llmq_doctor_command():
    """Test that llmq doctor command works."""
    result = subprocess.run([sys.executable, "-m", "cli.main", "doctor"], 
                          capture_output=True, text=True)
    assert result.returncode == 0
    assert "LLMQ Doctor - System Health Check" in result.stdout

def test_llmq_help_shows_all_commands():
    """Test that help shows all expected commands."""
    result = subprocess.run([sys.executable, "-m", "cli.main", "--help"], 
                          capture_output=True, text=True)
    assert result.returncode == 0
    
    expected_commands = ["eval", "providers", "dashboard", "init", "doctor", "runs", "compare", "settings"]
    for command in expected_commands:
        assert command in result.stdout

def test_llmq_init_help():
    """Test that init command help works."""
    result = subprocess.run([sys.executable, "-m", "cli.main", "init", "--help"], 
                          capture_output=True, text=True)
    assert result.returncode == 0
    assert "Initialize a new LLMQ project" in result.stdout