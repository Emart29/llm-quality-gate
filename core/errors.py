"""Domain errors used by LLMQ services."""


class LLMQError(Exception):
    """Base exception for user-facing LLMQ errors."""


class ConfigError(LLMQError):
    """Raised when configuration cannot be located or parsed."""

    exit_code = 2


class EvaluationError(LLMQError):
    """Raised when evaluation fails unexpectedly."""

    exit_code = 1
