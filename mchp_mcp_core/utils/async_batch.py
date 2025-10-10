"""
Async batch processing utilities for concurrent operations.

Provides efficient batch processing with configurable concurrency limits,
error handling, and progress tracking.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional, TypeVar

from mchp_mcp_core.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')
R = TypeVar('R')


async def process_batch_concurrent(
    items: list[T],
    process_func: Callable[[T], Any],
    max_concurrent: int = 5,
    return_exceptions: bool = True,
    show_progress: bool = True
) -> list[R]:
    """
    Process items concurrently with configurable concurrency limit.

    Args:
        items: List of items to process
        process_func: Async function to process each item
        max_concurrent: Maximum concurrent operations (default 5)
        return_exceptions: If True, exceptions are returned instead of raised
        show_progress: If True, shows progress bar (requires tqdm)

    Returns:
        List of results (or exceptions if return_exceptions=True)

    Example:
        >>> async def process_doc(path):
        ...     # Extract and embed document
        ...     chunks = await extract(path)
        ...     return await embed(chunks)
        >>>
        >>> docs = list(Path("./docs").rglob("*.pdf"))
        >>> results = await process_batch_concurrent(
        ...     docs,
        ...     process_doc,
        ...     max_concurrent=10
        ... )
        >>> print(f"Processed {len(results)} documents")
    """
    if not items:
        return []

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(item: T) -> R:
        async with semaphore:
            try:
                return await process_func(item)
            except Exception as e:
                logger.error(f"Error processing {item}: {e}")
                if return_exceptions:
                    return e
                raise

    tasks = [process_with_semaphore(item) for item in items]

    if show_progress:
        try:
            from tqdm.asyncio import tqdm_asyncio
            results = await tqdm_asyncio.gather(*tasks, desc="Processing batch")
        except ImportError:
            logger.debug("tqdm not available, processing without progress bar")
            results = await asyncio.gather(*tasks)
    else:
        results = await asyncio.gather(*tasks)

    return results


async def process_batch_chunked(
    items: list[T],
    process_func: Callable[[list[T]], Any],
    chunk_size: int = 10,
    show_progress: bool = True
) -> list[R]:
    """
    Process items in chunks (useful for batch APIs).

    Args:
        items: List of items to process
        process_func: Async function that processes a chunk of items
        chunk_size: Number of items per chunk
        show_progress: If True, shows progress bar

    Returns:
        List of results (flattened from all chunks)

    Example:
        >>> async def embed_batch(texts):
        ...     # Batch embedding API call
        ...     return await embedding_model.embed(texts, batch_size=32)
        >>>
        >>> texts = [chunk.content for chunk in chunks]
        >>> embeddings = await process_batch_chunked(
        ...     texts,
        ...     embed_batch,
        ...     chunk_size=32
        ... )
    """
    if not items:
        return []

    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    results = []
    if show_progress:
        try:
            from tqdm.asyncio import tqdm
            for chunk in tqdm(chunks, desc="Processing chunks"):
                chunk_result = await process_func(chunk)
                if isinstance(chunk_result, list):
                    results.extend(chunk_result)
                else:
                    results.append(chunk_result)
        except ImportError:
            for chunk in chunks:
                chunk_result = await process_func(chunk)
                if isinstance(chunk_result, list):
                    results.extend(chunk_result)
                else:
                    results.append(chunk_result)
    else:
        for chunk in chunks:
            chunk_result = await process_func(chunk)
            if isinstance(chunk_result, list):
                results.extend(chunk_result)
            else:
                results.append(chunk_result)

    return results


async def rate_limited_batch(
    items: list[T],
    process_func: Callable[[T], Any],
    rate_limit: int,
    period: float = 1.0
) -> list[R]:
    """
    Process items with rate limiting (X items per period).

    Args:
        items: List of items to process
        process_func: Async function to process each item
        rate_limit: Number of items allowed per period
        period: Time period in seconds (default 1.0)

    Returns:
        List of results

    Example:
        >>> # API allows 10 requests per second
        >>> results = await rate_limited_batch(
        ...     queries,
        ...     process_query,
        ...     rate_limit=10,
        ...     period=1.0
        ... )
    """
    if not items:
        return []

    results = []
    for i in range(0, len(items), rate_limit):
        batch = items[i:i + rate_limit]
        batch_results = await asyncio.gather(*[process_func(item) for item in batch])
        results.extend(batch_results)

        # Sleep if not last batch
        if i + rate_limit < len(items):
            await asyncio.sleep(period)

    return results
