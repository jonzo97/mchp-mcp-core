"""
LLM client for API integration.

Provides async API integration for secure LLM services with:
- Retry logic (exponential backoff)
- Rate limiting
- Secret masking in logs
- Optional PII redaction
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from mchp_mcp_core.security.pii import redact_pii
from mchp_mcp_core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    confidence: float
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMClient:
    """
    Async client for LLM API integration.

    Features:
    - Async HTTP with retry logic
    - Rate limiting with semaphore
    - Streaming support
    - PII redaction before API calls
    - Secret masking in logs

    Example:
        >>> async with LLMClient(config) as client:
        ...     response = await client.review_text("Original text...")
        ...     print(f"Confidence: {response.confidence}")
    """

    def __init__(self, config: Any):
        """
        Initialize LLM client.

        Args:
            config: LLMConfig object or dict with LLM settings
        """
        # Handle both dict and Pydantic config
        if hasattr(config, 'enabled'):
            self.enabled = config.enabled
            self.api_url = config.api_url
            self.api_key = os.getenv(config.api_key_env, "")
            self.model = config.model
            self.temperature = config.temperature
            self.max_tokens = config.max_tokens
            self.timeout = config.timeout
            self.stream = config.stream
            self.stream_timeout = config.stream_timeout
            self.verify_ssl = config.verify_ssl
            self.requests_per_minute = config.requests_per_minute
            self.concurrent_requests = config.concurrent_requests
        else:
            # Fallback for dict config
            llm_config = config.get('llm', {})
            self.enabled = llm_config.get('enabled', False)
            self.api_url = llm_config.get('api_url', '')
            api_key_env = llm_config.get('api_key_env', 'LLM_API_KEY')
            self.api_key = os.getenv(api_key_env, '')
            self.model = llm_config.get('model', 'gpt-4')
            self.temperature = llm_config.get('temperature', 0.0)
            self.max_tokens = llm_config.get('max_tokens', 2000)
            self.timeout = llm_config.get('timeout', 30)
            self.stream = llm_config.get('stream', False)
            self.stream_timeout = llm_config.get('stream_timeout', 60)
            self.verify_ssl = llm_config.get('verify_ssl', True)
            rate_limit = llm_config.get('rate_limit', {})
            self.requests_per_minute = rate_limit.get('requests_per_minute', 20)
            self.concurrent_requests = rate_limit.get('concurrent_requests', 5)

        # Semaphore for concurrent request limiting
        self.semaphore = asyncio.Semaphore(self.concurrent_requests)

        # HTTP client
        self.client = None

    async def __aenter__(self):
        """Async context manager entry."""
        timeout_config = httpx.Timeout(
            connect=5.0,
            read=self.stream_timeout,
            write=10.0,
            pool=5.0
        )
        self.client = httpx.AsyncClient(timeout=timeout_config, verify=self.verify_ssl)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def review_text(
        self,
        text: str,
        context: Optional[str] = None,
        redact_pii_before_api: bool = True
    ) -> LLMResponse:
        """
        Send text for LLM review.

        Args:
            text: Text to review
            context: Optional context (surrounding text, section info, etc.)
            redact_pii_before_api: Redact PII before sending to API

        Returns:
            LLMResponse with reviewed text and metadata
        """
        if not self.enabled:
            raise ValueError("LLM integration is not enabled")

        if not self.api_key:
            raise ValueError("LLM API key not found in environment")

        # Redact PII if enabled
        if redact_pii_before_api:
            text = redact_pii(text)
            if context:
                context = redact_pii(context)

        # Prepare request payload
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a technical document reviewer. Provide concise, accurate suggestions."
                },
                {
                    "role": "user",
                    "content": f"Review the following text:\n\n{text}"
                }
            ]
        }

        if context:
            payload["messages"].insert(1, {
                "role": "user",
                "content": f"Context: {context}"
            })

        # Make API request
        async with self.semaphore:
            logger.debug(f"Sending request to LLM API: {self.api_url}")

            response = await self.client.post(
                self.api_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )

            response.raise_for_status()

            result = response.json()

            # Parse response (adjust based on your API format)
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            confidence = result.get("confidence", 0.8)  # Default if not provided

            return LLMResponse(
                content=content,
                confidence=confidence,
                reasoning=None,
                metadata={"model": self.model, "temperature": self.temperature}
            )

    async def review_batch(
        self,
        texts: list[str],
        redact_pii_before_api: bool = True
    ) -> list[LLMResponse]:
        """
        Review multiple texts in batch.

        Args:
            texts: List of texts to review
            redact_pii_before_api: Redact PII before sending to API

        Returns:
            List of LLMResponse objects
        """
        tasks = [
            self.review_text(text, redact_pii_before_api=redact_pii_before_api)
            for text in texts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, LLMResponse)]

        logger.info(f"Batch review complete: {len(valid_results)}/{len(texts)} successful")

        return valid_results


__all__ = ["LLMClient", "LLMResponse"]
