# mchp-mcp-core

**Shared core library for Microchip MCP servers and RAG tools**

## Overview

`mchp-mcp-core` is a reusable Python library that provides common functionality for building MCP (Model Context Protocol) servers and RAG (Retrieval-Augmented Generation) systems at Microchip. It follows the principle that **MCP servers should be thin wrappers around reusable business logic**.

## Architecture

```
MCP Server (thin wrapper)
    ↓
mchp-mcp-core (reusable business logic)
    ↓
Data Layer (storage, retrieval)
```

## Features

### 📄 Document Extraction (`mchp_mcp_core.extractors`)
- **PDF Extraction**: PyMuPDF-based extraction with structure preservation
- **Table Extraction**: Multi-strategy table extraction using pdfplumber
- **PPTX Extraction**: PowerPoint slide parsing
- **Intelligent Chunking**: Fixed-size and semantic chunking strategies
- **Confidence Scoring**: Quality assessment for extracted content

### 🗄️ Vector Storage (`mchp_mcp_core.storage`)
- **Qdrant**: Production-ready vector store with hybrid search (BM25 + vector)
- **ChromaDB**: Development-friendly vector database (NEW in v0.1.1)
- **SQLite**: Async caching layer for state management (NEW in v0.1.1)
- **Manifest System**: Document versioning and ingestion tracking (NEW in v0.1.1)
- **Hybrid Storage**: Automatic fallback between vector and cache
- **Deduplication**: SHA-256 content hashing

### 🧠 Embeddings (`mchp_mcp_core.embeddings`)
- **sentence-transformers**: Clean wrapper for bge-small-en and similar models
- **Auto Device Detection**: CPU/CUDA/MPS support
- **Batch Processing**: Progress bars and efficient batching
- **Caching**: Optional disk caching for embeddings

### 🤖 LLM Integration (`mchp_mcp_core.llm`)
- **Async Client**: HTTP client with retry logic (tenacity)
- **Internal API Support**: Integration patterns for secure LLM services
- **Confidence Scoring**: Calibrated predictions
- **Streaming**: Support for streaming responses

### 🔒 Security (`mchp_mcp_core.security`)
- **PII Redaction**: Remove emails, phones, SSNs before API calls
- **Path Validation**: Prevent path traversal attacks
- **File Type Restrictions**: Whitelist-based file validation
- **Sandboxing**: Restrict operations to workspace directories

### 🛠️ Utilities (`mchp_mcp_core.utils`)
- **Configuration**: Pydantic Settings-based config management
- **Logging**: Structured logging with secret masking
- **Common Models**: Shared Pydantic schemas

## Installation

### From source (editable install)
```bash
cd /home/jorgill/mchp-mcp-core
pip install -e .
```

### With development dependencies
```bash
pip install -e ".[dev]"
```

## Usage

### Example: PDF Extraction
```python
from mchp_mcp_core.extractors import PDFExtractor

extractor = PDFExtractor(config)
chunks = extractor.extract_document("datasheet.pdf", "doc_id_123")
print(f"Extracted {len(chunks)} chunks")
```

### Example: Vector Storage (Qdrant)
```python
from mchp_mcp_core.storage import QdrantVectorStore, SearchQuery
from mchp_mcp_core.embeddings import EmbeddingModel

embedder = EmbeddingModel()
store = QdrantVectorStore(embedding_model=embedder)

# Add documents
store.add_documents(chunks)

# Hybrid search (BM25 + vector)
query = SearchQuery(query="FPGA clock frequency", top_k=5, hybrid=True)
results = store.search(query)
```

### Example: ChromaDB (Development)
```python
from mchp_mcp_core.storage import ChromaDBVectorStore, SearchQuery

# No server required - perfect for dev/prototyping
store = ChromaDBVectorStore(db_path="./chroma_data")

if store.is_available():
    store.add_documents(chunks)
    query = SearchQuery(query="memory specifications", top_k=5)
    results = store.search(query)
```

### Example: SQLite Caching
```python
from mchp_mcp_core.storage import SQLiteCache

async with SQLiteCache("./cache.db") as cache:
    # Cache chunks for fast exact retrieval
    await cache.insert_chunks(chunks)

    # Retrieve by document ID
    stored_chunks = await cache.get_chunks(doc_id="datasheet_001")

    # Store embeddings as fallback
    await cache.insert_embeddings([(chunk_id, vector)], model_name="bge-small")
```

### Example: Document Manifest
```python
from mchp_mcp_core.storage import ManifestRepository, DocumentManifest, ManifestStatus
from pathlib import Path

repo = ManifestRepository(Path("./manifest.db"))

# Track document version and processing state
manifest = DocumentManifest(
    doc_id="polarfire_datasheet",
    version="v2.1",
    checksum="abc123...",
    size_bytes=2048000,
    status=ManifestStatus.STAGED
)

entry = repo.upsert(manifest)
repo.update_status(entry.checksum, ManifestStatus.READY)

# Query by status
ready_docs = repo.list_by_status(ManifestStatus.READY)
```

### Example: LLM Client
```python
from mchp_mcp_core.llm import LLMClient

async with LLMClient(config) as client:
    response = await client.review_text(
        text="Original content...",
        context="Section 5.3 - Power Management"
    )
    print(f"Confidence: {response.confidence:.2%}")
```

## Projects Using This Library

- **mchp-fpga-rag**: FPGA documentation search and RAG system
- **datasheet-review**: Automated datasheet review and validation
- *(Future MCP servers)*

## Development

### Run Tests
```bash
pytest
```

### Code Quality
```bash
# Format code
black mchp_mcp_core/

# Lint
ruff check mchp_mcp_core/

# Type check
mypy mchp_mcp_core/
```

### Test Coverage
```bash
pytest --cov=mchp_mcp_core --cov-report=html
```

## Design Principles

1. **Pure Python**: No MCP dependencies in core library
2. **Testable**: All modules can be tested independently
3. **Reusable**: DRY - extract once, use everywhere
4. **Well-Typed**: Type hints throughout
5. **Documented**: Docstrings and examples

## Contributing

When adding new functionality:

1. **Check if it's domain-specific**:
   - If yes → belongs in domain library (e.g., `fpga_rag_core`)
   - If no → belongs here in `mchp-mcp-core`

2. **The 2+ rule**: If used by 2+ MCP servers, extract to core

3. **Keep it pure**: No MCP server code, no domain logic

4. **Test it**: Add unit tests for all new modules

## License

Internal use only. Not licensed for external distribution.

## Maintainer

Jonathan Orgill - Microchip FAE

---

**Generated:** 2025-10-09
**Version:** 0.1.1 (Phase 2A: Storage Completeness)

## Release Notes

### v0.1.1 (2025-10-09) - Phase 2A: Storage Completeness
- ✅ **ChromaDB Support**: Development-friendly vector store with graceful fallback
- ✅ **SQLite Caching**: Async state management and embedding fallback
- ✅ **Manifest System**: Document versioning with SQLModel
- ✅ **Enhanced Dependencies**: Added aiosqlite, sqlmodel
- ✅ **Documentation**: PROJECTS_MAP.md with tool selection intelligence

### v0.1.0 (2025-10-07) - Initial Release
- ✅ Core extractors (PDF, PPTX, tables, chunking)
- ✅ Qdrant vector store with hybrid search
- ✅ Embedding wrapper (sentence-transformers)
- ✅ LLM client with retry logic
- ✅ Security utilities (PII, validation)
- ✅ Pydantic-based configuration
