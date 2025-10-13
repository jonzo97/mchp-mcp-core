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
- **DOCX Extraction**: Microsoft Word document processing (NEW in v0.1.4)
- **Intelligent Chunking**: Fixed-size and semantic chunking strategies
- **Metadata Extraction**: Document type, product family, version, and date detection
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

### 📊 Analysis (`mchp_mcp_core.analysis`)
- **Terminology Consistency**: Detect variations and inconsistencies in technical terms
- **Abbreviation Expansion**: 70+ default technical abbreviations for improved search recall (NEW in v0.1.4)
- **Configurable Patterns**: Extensible regex patterns for domain-specific terminology
- **Severity Scoring**: Critical, high, medium, low issue classification
- **Brand Compliance**: Flag branded vs unbranded term mixing

### ✅ Validation (`mchp_mcp_core.validation`) (NEW in v0.1.3)
- **Semantic Completeness**: LLM-based claim extraction and evidence validation
- **Missing Evidence Detection**: Identify unsupported claims in documentation
- **Knowledge Base Integration**: Optional RAG enhancement with approved examples
- **Actionable Suggestions**: Generate specific recommendations for improvement

### 🛠️ Utilities (`mchp_mcp_core.utils`)
- **Configuration**: Pydantic Settings-based config management
- **Logging**: Structured logging with secret masking
- **Hashing**: File checksums and deduplication detection
- **Async Batch Processing**: Concurrent processing with semaphore control (NEW in v0.1.4)
- **Common Models**: Shared Pydantic schemas

### 🔄 Ingestion (`mchp_mcp_core.ingestion`) (NEW in v0.1.5)
- **Orchestration**: Multi-format batch ingestion with parallel processing
- **Manifest Tracking**: Checksum-based deduplication and status tracking
- **Progress Display**: Rich progress bars with ETA
- **Error Handling**: Graceful failure with detailed reporting
- **JSONL Export**: Standard corpus format for downstream processing
- **HTML Reports**: Visual summaries with metrics

### 📝 Output (`mchp_mcp_core.output`) (NEW in v0.1.6)
- **Markdown Generation**: Document reassembly with TOC and citations
- **Result Filtering**: Severity-based prioritization (CRITICAL → LOW)
- **Deduplication**: Similarity-based result filtering
- **Citation Formatting**: Standardized reference strings
- **JSON Export**: Programmatic result access

### 📈 Evaluation (`mchp_mcp_core.evaluation`) (NEW in v0.1.6)
- **Precision@K**: Standard IR metric (P@1, P@3, P@5)
- **Mean Reciprocal Rank**: MRR calculation
- **Recall@K**: Coverage metrics
- **Batch Evaluation**: Test query processing with progress tracking
- **HTML Reports**: Dark-themed metric dashboards
- **Latency Tracking**: Query performance monitoring

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

### Example: Metadata Extraction
```python
from mchp_mcp_core.extractors import MetadataExtractor, extract_metadata
from pathlib import Path

# Simple extraction
metadata = extract_metadata(
    Path("docs/PolarFire/PolarFire-FPGA-Datasheet-DS00003831.pdf"),
    first_page_text="PolarFire FPGA Datasheet\nRevision 5\nJanuary 2024",
    docs_root=Path("docs")
)

print(metadata["document_type"])   # "Datasheet"
print(metadata["product_family"])  # "PolarFire"
print(metadata["version"])         # "00003831"
print(metadata["document_date"])   # "2024-01"
print(metadata["category_tags"])   # ["PolarFire", "datasheet", "polarfire"]

# Custom patterns for your domain
custom_extractor = MetadataExtractor(
    product_patterns=[
        (r"widget[_\s]pro", "Widget Pro"),
        (r"gadget", "Gadget Series"),
    ]
)
```

### Example: File Hashing and Deduplication
```python
from mchp_mcp_core.utils import compute_checksum, find_duplicates
from pathlib import Path

# Compute checksum for a single file
checksum = compute_checksum("document.pdf")
print(f"SHA-256: {checksum}")

# Find duplicate files in a directory
all_pdfs = list(Path("./docs").rglob("*.pdf"))
duplicates = find_duplicates(all_pdfs)

for checksum, file_list in duplicates.items():
    print(f"Duplicate set ({len(file_list)} files):")
    for path in file_list:
        print(f"  - {path}")
```

### Example: Terminology Consistency Analysis
```python
from mchp_mcp_core.analysis import TerminologyAnalyzer

# Use default patterns (connectivity, peripherals, features, memory)
analyzer = TerminologyAnalyzer()
report = await analyzer.analyze_terminology(chunks)

print(f"Consistency rate: {report['consistency_rate']}%")
print(f"Issues found: {report['inconsistent_terms']}")

# Critical issues (e.g., brand compliance)
for issue in report['critical_issues']:
    print(f"❗ {issue['term']}: {issue['variations']}")
    print(f"   Recommend: {issue['recommended']}")

# Custom patterns for your domain
custom_patterns = {
    'protocols': [
        (r'TCP/IP', 'TCP/IP'),
        (r'tcpip', 'TCP/IP'),
        (r'Modbus', 'Modbus'),
    ]
}
analyzer = TerminologyAnalyzer(term_patterns=custom_patterns)
```

### Example: Semantic Completeness Validation
```python
from mchp_mcp_core.validation import CompletenessValidator
from mchp_mcp_core.llm import LLMClient

# Initialize with LLM client
llm = LLMClient(config)
validator = CompletenessValidator(llm_client=llm)

# Validate documentation completeness
report = await validator.validate_completeness(chunks)

print(f"Support rate: {report['support_rate']}%")
print(f"Claims: {report['total_claims']}")
print(f"Unsupported: {report['claims_unsupported']}")

# Critical issues (claims without evidence)
for issue in report['critical_issues']:
    print(f"❗ Claim: {issue['claim']}")
    print(f"   Section: {issue['section']} (p.{issue['page']})")
    print(f"   Missing: {', '.join(issue['missing_evidence'])}")
    print(f"   Suggestion: {issue['suggestion']}")

# With knowledge base (RAG enhancement)
validator = CompletenessValidator(
    llm_client=llm,
    knowledge_base=kb_manager  # Provides approved examples
)
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

### Example: Abbreviation Expansion
```python
from mchp_mcp_core.analysis import AbbreviationExpander, DEFAULT_ABBREVIATIONS

expander = AbbreviationExpander(DEFAULT_ABBREVIATIONS)

# Expand for better search recall
query = "Configure SPI bus and I2C interface"
expanded = expander.expand_query(query)
# → "Configure SPI Serial Peripheral Interface bus and I2C Inter-Integrated Circuit interface"

# 70+ defaults: SPI, I2C, UART, USB, WiFi, SRAM, ADC, PWM, GPIO, etc.
```

### Example: Ingestion Orchestrator
```python
from mchp_mcp_core.ingestion import IngestionOrchestrator
from mchp_mcp_core.storage import QdrantVectorStore, ManifestRepository
from mchp_mcp_core.embeddings import EmbeddingModel
from pathlib import Path

# Setup components
orchestrator = IngestionOrchestrator(
    vector_store=QdrantVectorStore(),
    embedding_model=EmbeddingModel(),
    manifest_repo=ManifestRepository(Path("./data/manifest.db")),
    max_concurrent=10
)

# Run complete pipeline
report = await orchestrator.run(
    directory="./docs",
    output_jsonl="./data/corpus.jsonl",
    report_html="./data/report.html",
    recursive=True
)

print(f"✅ Processed {report.total_files} files")
print(f"📊 Created {report.total_chunks} chunks")
print(f"⏱️  Duration: {report.duration_seconds:.1f}s")
print(f"✨ Success rate: {report.success_rate:.1f}%")
```

### Example: Markdown Generation
```python
from mchp_mcp_core.output import MarkdownGenerator

generator = MarkdownGenerator(include_toc=True, include_page_refs=True)

# Generate from search results
markdown = generator.generate_from_results(
    results=search_results,
    title="PolarFire FPGA Documentation",
    metadata={"product": "PolarFire", "version": "v2.0"}
)

with open("output.md", "w") as f:
    f.write(markdown)
```

### Example: Result Filtering
```python
from mchp_mcp_core.output import OutputFilter, filter_results_by_score

# Filter by severity
filter_obj = OutputFilter(verbosity='normal')
changes = [
    {'type': 'crossref', 'reason': 'broken reference', 'valid': False},
    {'type': 'style', 'reason': 'terminology inconsistency'}
]
filtered = filter_obj.filter_changes(changes)
print(filter_obj.format_summary(filtered))

# Filter search results by score
high_quality = filter_results_by_score(results, min_score=0.75, max_results=5)
```

### Example: Evaluation Metrics
```python
from mchp_mcp_core.evaluation import EvaluationMetrics

# Batch evaluation
evaluator = EvaluationMetrics(search_fn=vector_store.search)

test_queries = [
    {"query": "SPI configuration", "expected_doc": "Datasheet.pdf", "category": "peripherals"},
    {"query": "power modes", "expected_doc": "UserGuide.pdf", "category": "power"}
]

report = evaluator.evaluate_queries(test_queries)

print(f"P@1: {report.p_at_1:.1%}")
print(f"P@3: {report.p_at_3:.1%}")  # Target: 70%+
print(f"MRR: {report.mrr:.3f}")

# Generate HTML report
html = evaluator.generate_html_report(report, title="Quality Assessment")
Path("eval_report.html").write_text(html)
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

**Generated:** 2025-10-10
**Version:** 0.1.6 (Phase 2E Complete: C→B→A→D)

## Release Notes

### v0.1.6 (2025-10-10) - Phases A+D: Output & Evaluation
- ✅ **Markdown Generation**: Document reassembly with TOC and citations
- ✅ **Output Filtering**: Severity-based prioritization (CRITICAL/HIGH/MEDIUM/LOW)
- ✅ **Result Deduplication**: Similarity-based filtering
- ✅ **Precision@K Metrics**: P@1, P@3, P@5 evaluation
- ✅ **Mean Reciprocal Rank**: MRR calculation
- ✅ **Batch Evaluation**: Test query processing with progress tracking
- ✅ **HTML Reports**: Dark-themed evaluation dashboards
- ✅ **Latency Tracking**: Query performance monitoring

### v0.1.5 (2025-10-10) - Phase B: Ingestion Orchestration
- ✅ **Multi-Format Orchestrator**: Unified ingestion for PDF/PPTX/DOCX
- ✅ **Parallel Processing**: Configurable concurrency with semaphore control
- ✅ **Manifest Integration**: Checksum-based deduplication and status tracking
- ✅ **Rich Progress Bars**: Visual feedback with ETA
- ✅ **Error Handling**: Graceful failure with detailed reporting
- ✅ **JSONL Export**: Standard corpus format
- ✅ **HTML Reports**: Visual ingestion summaries

### v0.1.4 (2025-10-10) - Phase C: Quick Wins
- ✅ **DOCX Extraction**: Microsoft Word document support
- ✅ **Abbreviation Expansion**: 70+ technical abbreviations (SPI, I2C, UART, etc.)
- ✅ **Async Batch Processing**: Concurrent processing with semaphore control
- ✅ **Rate Limiting**: Token bucket pattern for API calls
- ✅ **Enhanced Dependencies**: Added python-docx, rich

### v0.1.3 (2025-10-09) - Phase 2C: Analysis & Validation
- ✅ **Terminology Consistency Analysis**: Detect term variations and inconsistencies
- ✅ **Configurable Term Patterns**: Extensible for domain-specific terminology
- ✅ **Semantic Completeness Validation**: LLM-based claim extraction and evidence checking
- ✅ **Knowledge Base Integration**: Optional RAG enhancement for suggestions
- ✅ **Severity Classification**: Critical, high, medium, low issue levels
- ✅ **Actionable Reporting**: Specific recommendations for documentation improvement

### v0.1.2 (2025-10-09) - Phase 2B: Metadata & Hashing
- ✅ **Metadata Extraction**: Document type, product family, version, and date detection
- ✅ **Configurable Patterns**: Extensible regex patterns for domain-specific extraction
- ✅ **File Hashing**: SHA-256 checksums for deduplication and integrity verification
- ✅ **Duplicate Detection**: Find duplicate files by content hash
- ✅ **Batch Processing**: Process multiple documents/files efficiently
- ✅ **Enhanced Documentation**: Usage examples for metadata and hashing modules

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
