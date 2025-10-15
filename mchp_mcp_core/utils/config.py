"""
Configuration management using Pydantic Settings.

Provides base configuration classes that can be extended by specific MCP servers.
Supports environment variables, .env files, and programmatic configuration.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMTableConfig(BaseSettings):
    """Configuration for LLM-based table extraction (vision models)."""

    # Enable/disable
    enabled: bool = Field(default=False, description="Enable LLM table extraction")

    # Privacy controls
    local_only: bool = Field(default=False, description="Use only local models (no cloud)")
    allow_cloud: bool = Field(default=False, description="Allow cloud API calls")
    redact_pii_before_llm: bool = Field(default=True, description="Redact PII before LLM")

    # Provider selection
    provider: str = Field(default="openai", description="LLM provider: openai, anthropic, ollama")
    model: str = Field(default="gpt-4o", description="Vision model name")
    api_key_env: str = Field(default="OPENAI_API_KEY", description="Environment variable for API key")
    api_url: Optional[str] = Field(None, description="Custom API URL (for Ollama or self-hosted)")

    # Model parameters
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Temperature")
    max_tokens: int = Field(default=4000, ge=1, description="Max tokens in response")
    timeout: int = Field(default=60, ge=1, description="Request timeout (seconds)")

    # Fallback strategy
    use_as_fallback: bool = Field(default=True, description="Use LLM only when confidence < threshold")
    fallback_threshold: float = Field(default=0.70, ge=0.0, le=1.0, description="Confidence threshold")
    use_for_no_results: bool = Field(default=True, description="Use LLM if no tables found")

    # Limits
    max_image_size_mb: float = Field(default=5.0, ge=0.1, description="Max image size for LLM")

    model_config = SettingsConfigDict(
        env_prefix="LLM_TABLE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )


class ExtractionConfig(BaseSettings):
    """Configuration for document extraction."""

    # Chunking settings
    chunk_size: int = Field(default=1500, description="Target chunk size in characters")
    overlap: int = Field(default=200, description="Overlap between chunks")
    min_chunk_size: int = Field(default=500, description="Minimum chunk size (semantic)")
    max_chunk_size: int = Field(default=2500, description="Maximum chunk size (semantic)")

    # Chunking strategy
    chunking_strategy: str = Field(
        default="fixed",
        description="Chunking strategy: 'fixed' or 'semantic'"
    )

    # Feature flags
    preserve_sections: bool = Field(default=True, description="Preserve section hierarchy")
    extract_images: bool = Field(default=True, description="Extract images/figures")
    extract_tables: bool = Field(default=True, description="Extract tables")

    # LLM table extraction
    llm_tables: LLMTableConfig = Field(default_factory=LLMTableConfig, description="LLM table extraction settings")

    model_config = SettingsConfigDict(
        env_prefix="EXTRACTION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )


class StorageConfig(BaseSettings):
    """Configuration for vector storage."""

    # Qdrant settings
    qdrant_host: str = Field(default="localhost", description="Qdrant server host")
    qdrant_port: int = Field(default=6333, description="Qdrant server port")
    qdrant_on_disk: bool = Field(default=True, description="Store vectors on disk")
    collection_name: str = Field(default="documents", description="Collection name")
    hybrid_search: bool = Field(default=True, description="Enable hybrid search (BM25 + vector)")

    # ChromaDB settings
    chromadb_path: Optional[str] = Field(None, description="ChromaDB persistence directory")

    # SQLite settings
    sqlite_path: Optional[str] = Field(None, description="SQLite database path")

    model_config = SettingsConfigDict(
        env_prefix="STORAGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )


class EmbeddingConfig(BaseSettings):
    """Configuration for embedding models."""

    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model name"
    )
    embedding_device: Optional[str] = Field(
        None,
        description="Device: 'cpu', 'cuda', 'mps' (auto-detect if None)"
    )
    embedding_batch_size: int = Field(default=32, description="Batch size for encoding")
    embedding_cache_dir: Optional[Path] = Field(None, description="Embedding cache directory")
    hf_home: Optional[Path] = Field(None, description="HuggingFace cache directory")

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )


class LLMConfig(BaseSettings):
    """Configuration for LLM API integration."""

    enabled: bool = Field(default=False, description="Enable LLM integration")
    api_url: str = Field(default="", description="LLM API endpoint URL")
    api_key_env: str = Field(default="LLM_API_KEY", description="Environment variable for API key")
    model: str = Field(default="gpt-4", description="Model name")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Temperature")
    max_tokens: int = Field(default=2000, ge=1, description="Max tokens in response")
    timeout: int = Field(default=30, ge=1, description="Request timeout (seconds)")
    stream: bool = Field(default=False, description="Enable streaming responses")
    stream_timeout: int = Field(default=60, description="Streaming timeout (seconds)")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")

    # Rate limiting
    requests_per_minute: int = Field(default=20, ge=1, description="Max requests per minute")
    concurrent_requests: int = Field(default=5, ge=1, description="Max concurrent requests")

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )


class SecurityConfig(BaseSettings):
    """Configuration for security features."""

    enable_pii_redaction: bool = Field(default=True, description="Enable PII redaction")
    workspace_dir: Path = Field(default=Path("./data"), description="Workspace directory")
    allowed_file_types: list[str] = Field(
        default=[".pdf", ".pptx", ".docx", ".txt"],
        description="Allowed file extensions"
    )
    max_file_size_mb: int = Field(default=50, ge=1, description="Max file size in MB")
    allow_cloud: bool = Field(default=False, description="Allow external API calls")

    model_config = SettingsConfigDict(
        env_prefix="SECURITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )


class MCPCoreSettings(BaseSettings):
    """
    Unified configuration for mchp-mcp-core.

    Combines all sub-configurations into a single settings object.
    Individual MCP servers can extend this or use specific sub-configs.
    """

    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # General settings
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )


# Singleton instance
_settings: Optional[MCPCoreSettings] = None


def get_settings() -> MCPCoreSettings:
    """Get global settings instance (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = MCPCoreSettings()
    return _settings


def load_config_from_dict(config_dict: Dict[str, Any]) -> MCPCoreSettings:
    """
    Load configuration from a dictionary (for backward compatibility).

    This allows legacy code using dict-based configs to work with new Pydantic settings.
    """
    # Extract nested config sections
    extraction_config = ExtractionConfig(**config_dict.get("document", {}))
    storage_config = StorageConfig(**config_dict.get("storage", {}))
    embedding_config = EmbeddingConfig(**config_dict.get("embedding", {}))
    llm_config = LLMConfig(**config_dict.get("llm", {}))
    security_config = SecurityConfig(**config_dict.get("security", {}))

    return MCPCoreSettings(
        extraction=extraction_config,
        storage=storage_config,
        embedding=embedding_config,
        llm=llm_config,
        security=security_config
    )
