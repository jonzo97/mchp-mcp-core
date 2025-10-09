"""
Path validation and filename sanitization utilities.

Provides defensive security measures to prevent:
- Directory traversal attacks
- Invalid filenames
- Path manipulation
"""

import re
from pathlib import Path


def validate_path(path: Path, workspace: Path) -> bool:
    """
    Validate that a path is within the allowed workspace.

    Prevents directory traversal attacks.

    Args:
        path: Path to validate
        workspace: Allowed workspace directory

    Returns:
        True if path is within workspace, False otherwise

    Example:
        >>> workspace = Path("/home/user/project")
        >>> validate_path(Path("/home/user/project/data/file.txt"), workspace)
        True
        >>> validate_path(Path("/etc/passwd"), workspace)
        False
    """
    try:
        # Resolve both paths to absolute paths
        resolved_path = path.resolve()
        resolved_workspace = workspace.resolve()

        # Check if path is relative to workspace
        return resolved_path.is_relative_to(resolved_workspace)

    except (ValueError, OSError):
        # If resolution fails, deny access
        return False


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a filename to prevent path traversal and invalid characters.

    Args:
        filename: Original filename
        max_length: Maximum filename length (default: 255)

    Returns:
        Sanitized filename

    Example:
        >>> sanitize_filename("../../etc/passwd")
        'etc_passwd'
        >>> sanitize_filename("my<file>name.txt")
        'my_file_name.txt'
    """
    # Remove path separators
    sanitized = filename.replace("/", "_").replace("\\", "_")

    # Remove or replace invalid characters
    # Keep alphanumeric, dash, underscore, period
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', sanitized)

    # Remove leading/trailing dots and underscores
    sanitized = sanitized.strip("._")

    # Truncate if too long
    if len(sanitized) > max_length:
        # Preserve extension if present
        if "." in sanitized:
            name, ext = sanitized.rsplit(".", 1)
            max_name_length = max_length - len(ext) - 1
            sanitized = name[:max_name_length] + "." + ext
        else:
            sanitized = sanitized[:max_length]

    # Ensure not empty
    if not sanitized:
        sanitized = "unnamed_file"

    return sanitized


def validate_file_type(filepath: Path, allowed_extensions: list[str]) -> bool:
    """
    Validate that a file has an allowed extension.

    Args:
        filepath: Path to file
        allowed_extensions: List of allowed extensions (e.g., ['.pdf', '.txt'])

    Returns:
        True if file type is allowed, False otherwise

    Example:
        >>> validate_file_type(Path("document.pdf"), ['.pdf', '.docx'])
        True
        >>> validate_file_type(Path("script.sh"), ['.pdf', '.docx'])
        False
    """
    return filepath.suffix.lower() in [ext.lower() for ext in allowed_extensions]


def validate_file_size(filepath: Path, max_size_mb: int) -> bool:
    """
    Validate that a file is not larger than the maximum allowed size.

    Args:
        filepath: Path to file
        max_size_mb: Maximum file size in megabytes

    Returns:
        True if file size is within limit, False otherwise

    Example:
        >>> validate_file_size(Path("large_file.pdf"), max_size_mb=50)
        True  # if file is < 50MB
    """
    try:
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        return file_size_mb <= max_size_mb
    except OSError:
        return False


__all__ = [
    "validate_path",
    "sanitize_filename",
    "validate_file_type",
    "validate_file_size"
]
