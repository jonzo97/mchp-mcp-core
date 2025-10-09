"""
LLM integration module.

Provides LLM API clients and integration patterns:
- Async HTTP client with retry logic
- Internal API support
- Confidence scoring
- Streaming support
"""

from mchp_mcp_core.llm.client import LLMClient, LLMResponse

__all__ = ["LLMClient", "LLMResponse"]
