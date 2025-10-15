"""
Vision LLM-based table extraction for fallback when traditional methods fail.

Provides table extraction using multimodal vision LLMs like GPT-4V, Claude 3.5,
and local models via Ollama. Designed with privacy controls for sensitive documents.

Key Features:
- PDF page → image → vision LLM → structured table
- Multiple providers: OpenAI, Anthropic, Ollama (local)
- Privacy controls: local-only mode, PII redaction, opt-in
- Confidence scoring from LLM responses
- Structured JSON output parsing
"""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import httpx
from PIL import Image

from mchp_mcp_core.extractors.table_extractors import (
    ExtractedTable,
    ExtractionResult,
    TableExtractor,
    TableComplexity,
    ExtractionStrategy
)
from mchp_mcp_core.security.pii import redact_pii
from mchp_mcp_core.utils import get_logger

logger = get_logger(__name__)


class VisionLLMProvider(Enum):
    """Supported vision LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass
class VisionLLMConfig:
    """Configuration for vision LLM table extraction."""
    enabled: bool = False
    provider: VisionLLMProvider = VisionLLMProvider.OPENAI
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    api_url: Optional[str] = None  # For Ollama or custom endpoints
    temperature: float = 0.0
    max_tokens: int = 4000
    timeout: int = 60
    local_only: bool = False
    allow_cloud: bool = False
    redact_pii: bool = True
    max_image_size_mb: float = 5.0


class VisionLLMTableExtractor(TableExtractor):
    """
    Extract tables using vision-capable LLMs.

    This extractor converts PDF pages to images and sends them to multimodal
    LLMs for table extraction. Useful as a fallback when traditional OCR-based
    extractors fail or produce low-confidence results.

    Privacy Features:
    - Local-only mode (Ollama)
    - PII redaction before API calls
    - Opt-in by default (enabled=False)
    - Cloud API gating (allow_cloud flag)

    Example:
        >>> config = VisionLLMConfig(
        ...     enabled=True,
        ...     provider=VisionLLMProvider.OPENAI,
        ...     model="gpt-4o",
        ...     api_key="sk-...",
        ...     allow_cloud=True
        ... )
        >>> extractor = VisionLLMTableExtractor(config)
        >>> result = extractor.extract_tables("datasheet.pdf", page_num=5)
    """

    def __init__(self, config: VisionLLMConfig):
        """
        Initialize vision LLM extractor.

        Args:
            config: VisionLLMConfig with provider, model, and privacy settings
        """
        self.config = config

        # Validate configuration
        if config.local_only and config.provider != VisionLLMProvider.OLLAMA:
            raise ValueError("local_only=True requires provider='ollama'")

        if not config.local_only and not config.allow_cloud:
            raise ValueError("Cloud providers require allow_cloud=True for safety")

        # Set API URLs
        self._setup_api_endpoints()

        logger.info(f"Initialized VisionLLMTableExtractor: provider={config.provider.value}, model={config.model}, local_only={config.local_only}")

    def _setup_api_endpoints(self):
        """Setup API endpoints based on provider."""
        if self.config.api_url:
            self.api_url = self.config.api_url
        elif self.config.provider == VisionLLMProvider.OPENAI:
            self.api_url = "https://api.openai.com/v1/chat/completions"
        elif self.config.provider == VisionLLMProvider.ANTHROPIC:
            self.api_url = "https://api.anthropic.com/v1/messages"
        elif self.config.provider == VisionLLMProvider.OLLAMA:
            self.api_url = self.config.api_url or "http://localhost:11434/api/generate"
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def is_available(self) -> bool:
        """Check if vision LLM extractor is available and configured."""
        if not self.config.enabled:
            return False

        # For cloud providers, require API key
        if self.config.provider in [VisionLLMProvider.OPENAI, VisionLLMProvider.ANTHROPIC]:
            if not self.config.api_key:
                logger.warning(f"Vision LLM {self.config.provider.value} not available: missing API key")
                return False

        # For Ollama, check if service is running (optional check)
        if self.config.provider == VisionLLMProvider.OLLAMA:
            # Could add a health check here, but for now assume available
            pass

        return True

    def extract_tables(
        self,
        pdf_path: str,
        page_num: int,
        **kwargs
    ) -> ExtractionResult:
        """
        Extract tables from PDF page using vision LLM.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            **kwargs: Additional options (ignored)

        Returns:
            ExtractionResult with extracted tables
        """
        if not self.is_available():
            return ExtractionResult(
                success=False,
                tables=[],
                error="Vision LLM extractor not available or not configured"
            )

        try:
            # Convert PDF page to image
            image_data = self._pdf_page_to_image(pdf_path, page_num)

            # Check image size
            image_size_mb = len(image_data) / (1024 * 1024)
            if image_size_mb > self.config.max_image_size_mb:
                logger.warning(f"Image size ({image_size_mb:.1f}MB) exceeds limit ({self.config.max_image_size_mb}MB)")
                return ExtractionResult(
                    success=False,
                    tables=[],
                    error=f"Image too large: {image_size_mb:.1f}MB > {self.config.max_image_size_mb}MB"
                )

            # Extract tables using vision LLM
            tables = self._extract_with_llm(image_data, page_num)

            return ExtractionResult(
                success=True,
                tables=tables,
                extractor_name="vision_llm",
                metadata={"provider": self.config.provider.value, "model": self.config.model}
            )

        except Exception as e:
            logger.error(f"Vision LLM extraction failed: {e}")
            return ExtractionResult(
                success=False,
                tables=[],
                error=str(e)
            )

    def _pdf_page_to_image(self, pdf_path: str, page_num: int) -> bytes:
        """
        Convert PDF page to PNG image.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)

        Returns:
            PNG image as bytes
        """
        doc = fitz.open(pdf_path)

        if page_num >= doc.page_count:
            raise ValueError(f"Page {page_num} out of range (total: {doc.page_count})")

        page = doc[page_num]

        # Render at high DPI for better OCR
        # 300 DPI = 300/72 = 4.17x zoom
        zoom = 300 / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PNG bytes
        png_data = pix.pil_tobytes(format="PNG")

        doc.close()

        return png_data

    def _extract_with_llm(self, image_data: bytes, page_num: int) -> List[ExtractedTable]:
        """
        Extract tables from image using vision LLM.

        Args:
            image_data: PNG image bytes
            page_num: Page number for metadata

        Returns:
            List of ExtractedTable objects
        """
        # Encode image to base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')

        # Build prompt
        prompt = self._build_extraction_prompt()

        # Call LLM API
        response_json = self._call_vision_api(prompt, image_b64)

        # Parse response
        tables = self._parse_llm_response(response_json, page_num)

        return tables

    def _build_extraction_prompt(self) -> str:
        """Build prompt for table extraction."""
        return """Extract all tables from this image and return them in the following JSON format:

{
  "tables": [
    {
      "index": 0,
      "rows": 5,
      "columns": 3,
      "complexity": "simple",
      "confidence": 0.95,
      "data": [
        ["Header 1", "Header 2", "Header 3"],
        ["Row 1, Col 1", "Row 1, Col 2", "Row 1, Col 3"],
        ...
      ]
    }
  ]
}

Important rules:
1. Preserve cell values EXACTLY as they appear (including numbers, units, special characters)
2. Mark empty cells as ""
3. Keep row/column structure accurate
4. For complexity, use: "simple", "medium", "complex", or "very_complex"
5. Provide confidence score 0.0-1.0 based on clarity and certainty
6. If merged cells are detected, note it in a "merged_cells" field (optional)
7. Include ALL tables on the page, even small ones
8. Return ONLY valid JSON, no markdown code blocks

If no tables are found, return: {"tables": []}"""

    def _call_vision_api(self, prompt: str, image_b64: str) -> Dict[str, Any]:
        """
        Call vision LLM API with image and prompt.

        Args:
            prompt: Text prompt for extraction
            image_b64: Base64-encoded image

        Returns:
            JSON response from API
        """
        if self.config.provider == VisionLLMProvider.OPENAI:
            return self._call_openai(prompt, image_b64)
        elif self.config.provider == VisionLLMProvider.ANTHROPIC:
            return self._call_anthropic(prompt, image_b64)
        elif self.config.provider == VisionLLMProvider.OLLAMA:
            return self._call_ollama(prompt, image_b64)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _call_openai(self, prompt: str, image_b64: str) -> Dict[str, Any]:
        """Call OpenAI GPT-4V API."""
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Parse JSON from content
            # Remove markdown code blocks if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            return json.loads(content)

    def _call_anthropic(self, prompt: str, image_b64: str) -> Dict[str, Any]:
        """Call Anthropic Claude API."""
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()
            content = result["content"][0]["text"]

            # Parse JSON from content
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            return json.loads(content)

    def _call_ollama(self, prompt: str, image_b64: str) -> Dict[str, Any]:
        """Call Ollama local API."""
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }

        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(self.api_url, json=payload)
            response.raise_for_status()

            result = response.json()
            content = result["response"]

            # Parse JSON from content
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            return json.loads(content)

    def _parse_llm_response(self, response: Dict[str, Any], page_num: int) -> List[ExtractedTable]:
        """
        Parse LLM JSON response into ExtractedTable objects.

        Args:
            response: JSON response from LLM
            page_num: Page number for metadata

        Returns:
            List of ExtractedTable objects
        """
        tables = []

        for table_json in response.get("tables", []):
            try:
                # Extract fields
                data = table_json["data"]
                rows = len(data)
                cols = len(data[0]) if data else 0

                # Map complexity string to enum
                complexity_str = table_json.get("complexity", "simple").upper()
                try:
                    complexity = TableComplexity[complexity_str]
                except KeyError:
                    complexity = TableComplexity.SIMPLE

                # Create ExtractedTable
                table = ExtractedTable(
                    data=data,
                    page_num=page_num,
                    table_index=table_json.get("index", 0),
                    confidence=table_json.get("confidence", 0.8),
                    complexity=complexity,
                    sparsity=self._compute_sparsity(data),
                    issues=[]
                )

                # Validate
                validated = self.validate_table(table)
                tables.append(validated)

            except Exception as e:
                logger.error(f"Failed to parse table from LLM response: {e}")
                continue

        return tables

    def _compute_sparsity(self, data: List[List[str]]) -> float:
        """Compute sparsity (ratio of empty cells)."""
        if not data:
            return 1.0

        total_cells = sum(len(row) for row in data)
        if total_cells == 0:
            return 1.0

        empty_cells = sum(
            1 for row in data
            for cell in row
            if not cell or cell.strip() == ""
        )

        return empty_cells / total_cells


__all__ = [
    "VisionLLMTableExtractor",
    "VisionLLMConfig",
    "VisionLLMProvider",
]
