"""Project and configuration discovery helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.errors import ConfigError

DEFAULT_CONFIG_NAME = "llmq.yaml"
PROJECT_MARKERS = (DEFAULT_CONFIG_NAME, "pyproject.toml", ".git")


def find_project_root(start_dir: Optional[Path] = None) -> Path:
    """Find project root by scanning upward for known project markers."""
    current = (start_dir or Path.cwd()).resolve()

    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate

    return current


def resolve_config_path(config_path: Optional[str] = None, start_dir: Optional[Path] = None) -> Path:
    """Resolve a config path from explicit input or project root discovery."""
    if config_path:
        provided = Path(config_path).expanduser()
        if not provided.is_absolute():
            provided = (start_dir or Path.cwd()) / provided
        provided = provided.resolve()
        if not provided.exists():
            raise ConfigError(f"Config file not found: {provided}")
        return provided

    project_root = find_project_root(start_dir)
    config_file = project_root / DEFAULT_CONFIG_NAME
    if not config_file.exists():
        raise ConfigError("No llmq.yaml found. Run `llmq init`.")
    return config_file
