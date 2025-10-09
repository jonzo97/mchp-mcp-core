"""
File hashing utilities for checksums and deduplication.

Provides SHA-256 and other hash algorithms for file content verification
and duplicate detection across document processing pipelines.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

from mchp_mcp_core.utils.logger import get_logger

logger = get_logger(__name__)


def compute_checksum(
    path: Union[Path, str],
    algorithm: str = "sha256",
    chunk_size: int = 8192
) -> str:
    """
    Compute the hex digest checksum for a file.

    Reads file in chunks to handle large files efficiently without
    loading the entire contents into memory.

    Args:
        path: Path to file to hash
        algorithm: Hash algorithm (sha256, md5, sha1, etc.)
        chunk_size: Number of bytes to read per chunk (default 8192)

    Returns:
        Hex digest string of the file hash

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If algorithm is not supported

    Example:
        >>> checksum = compute_checksum("document.pdf")
        >>> print(checksum)  # "a3f2b1c..."
        >>> # Verify two files are identical
        >>> hash1 = compute_checksum("file1.pdf")
        >>> hash2 = compute_checksum("file2.pdf")
        >>> if hash1 == hash2:
        ...     print("Files are identical")
    """
    path = Path(path) if isinstance(path, str) else path

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    logger.debug(f"Computing {algorithm} checksum for: {path.name}")

    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            hasher.update(chunk)

    digest = hasher.hexdigest()
    logger.debug(f"Checksum: {digest[:16]}... ({algorithm})")

    return digest


def compute_checksums(
    paths: list[Union[Path, str]],
    algorithm: str = "sha256",
    chunk_size: int = 8192
) -> dict[Path, str]:
    """
    Compute checksums for multiple files.

    Args:
        paths: List of file paths to hash
        algorithm: Hash algorithm (default sha256)
        chunk_size: Bytes per chunk (default 8192)

    Returns:
        Dict mapping Path to checksum hex digest

    Example:
        >>> paths = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        >>> checksums = compute_checksums(paths)
        >>> for path, checksum in checksums.items():
        ...     print(f"{path.name}: {checksum[:16]}...")
    """
    results = {}
    for path_item in paths:
        path = Path(path_item) if isinstance(path_item, str) else path_item
        try:
            results[path] = compute_checksum(path, algorithm, chunk_size)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Failed to compute checksum for {path}: {e}")
            continue

    return results


def find_duplicates(
    paths: list[Union[Path, str]],
    algorithm: str = "sha256"
) -> dict[str, list[Path]]:
    """
    Find duplicate files by content hash.

    Groups files with identical content together. Useful for deduplication
    before ingestion or identifying redundant documents.

    Args:
        paths: List of file paths to check
        algorithm: Hash algorithm (default sha256)

    Returns:
        Dict mapping checksum to list of paths with that checksum.
        Only returns entries with 2+ paths (duplicates).

    Example:
        >>> paths = list(Path("./docs").rglob("*.pdf"))
        >>> duplicates = find_duplicates(paths)
        >>> for checksum, file_list in duplicates.items():
        ...     print(f"Duplicate set ({len(file_list)} files):")
        ...     for path in file_list:
        ...         print(f"  - {path}")
    """
    checksums = compute_checksums(paths, algorithm)

    # Group paths by checksum
    groups: dict[str, list[Path]] = {}
    for path, checksum in checksums.items():
        if checksum not in groups:
            groups[checksum] = []
        groups[checksum].append(path)

    # Only return groups with duplicates (2+ files)
    duplicates = {
        checksum: file_list
        for checksum, file_list in groups.items()
        if len(file_list) > 1
    }

    if duplicates:
        logger.info(f"Found {len(duplicates)} duplicate sets across {sum(len(v) for v in duplicates.values())} files")
    else:
        logger.info("No duplicate files found")

    return duplicates
