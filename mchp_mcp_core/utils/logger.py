"""
Structured logging with secret masking.

Provides a standardized logger for all mchp-mcp-core modules with:
- Structured output (timestamps, log levels, module names)
- Secret masking (API keys, tokens, passwords)
- Configurable log levels
- File and console output support
"""

import logging
import re
import sys
from pathlib import Path
from typing import Optional


# Patterns for secret masking
SECRET_PATTERNS = [
    (re.compile(r'(api[_-]?key["\']?\s*[:=]\s*["\']?)([^"\']+)(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(token["\']?\s*[:=]\s*["\']?)([^"\']+)(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(password["\']?\s*[:=]\s*["\']?)([^"\']+)(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(secret["\']?\s*[:=]\s*["\']?)([^"\']+)(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(authorization["\']?\s*[:=]\s*["\']?)([^"\']+)(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
]


class SecretMaskingFormatter(logging.Formatter):
    """
    Logging formatter that masks secrets in log messages.

    Redacts API keys, tokens, passwords, and other sensitive data.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record and mask secrets."""
        # Get original formatted message
        original = super().format(record)

        # Apply secret masking patterns
        masked = original
        for pattern, replacement in SECRET_PATTERNS:
            masked = pattern.sub(replacement, masked)

        return masked


def setup_logger(
    name: str = "mchp_mcp_core",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    mask_secrets: bool = True
) -> logging.Logger:
    """
    Set up a structured logger with optional secret masking.

    Args:
        name: Logger name (typically module name)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        mask_secrets: Enable secret masking in log messages

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("my_module", level="DEBUG")
        >>> logger.info("Processing document")
        >>> logger.warning("API key: sk_test_1234", extra={"api_key": "***"})
    """
    logger = logging.getLogger(name)

    # Set log level
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatter
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if mask_secrets:
        formatter = SecretMaskingFormatter(log_format)
    else:
        formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with default configuration.

    Convenience function for getting loggers in modules.
    Uses environment variable LOG_LEVEL if set.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> from mchp_mcp_core.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    import os
    level = os.getenv("LOG_LEVEL", "INFO")
    return setup_logger(name, level=level)


# Module-level logger for mchp_mcp_core
logger = get_logger("mchp_mcp_core")


__all__ = ["setup_logger", "get_logger", "logger", "SecretMaskingFormatter"]
