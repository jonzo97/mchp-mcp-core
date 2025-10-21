# mchp-mcp-core Architecture

**Last Updated**: 2025-10-16 (Phase 3A completion)

This document describes the current architecture of `mchp-mcp-core`. **This file must be updated whenever code changes are made** - see `claude.md` for update instructions.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server (thin wrapper)                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                   mchp-mcp-core Library                      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Extractors  │  │  Embeddings  │  │     LLM      │      │
│  │              │  │              │  │              │      │
│  │ PDF, Tables, │  │ sentence-    │  │ HTTP client  │      │
│  │ PPTX, DOCX   │  │ transformers │  │ w/ retry     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                           │                                 │
│                           ↓                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Storage    │  │   Security   │  │    Utils     │      │
│  │              │  │              │  │              │      │
│  │ Qdrant, DB   │  │ PII, Paths   │  │ Config, Log  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              Data Layer (Vector DB, Cache, Files)            │
└─────────────────────────────────────────────────────────────┘
```

**Design Principle**: MCP servers should be thin wrappers around reusable business logic. All core functionality lives in `mchp-mcp-core` for reuse across projects.

---

## Module Architecture

### 1. Extractors (`mchp_mcp_core/extractors/`)

**Purpose**: Extract structured data from various document formats.

**Core Modules**:

#### Document Extractors
- `pdf.py` - PyMuPDF-based PDF extraction with structure preservation
- `pptx.py` - PowerPoint slide parsing using python-pptx
- `docx.py` - Word document parsing using python-docx
- `metadata.py` - Document metadata extraction (type, product, version, date)

#### Table Extraction (Simple)
- `tables.py` - Legacy multi-strategy table extraction
- `chunking.py` - Intelligent text chunking (fixed-size and semantic)

#### Table Extraction (Advanced) - **Phase 1-3A Complete** ✅
- `table_extractors.py` - Base extractor interface + 3 implementations:
  - `PdfPlumberExtractor` - Text-based extraction (multiple strategies)
  - `CamelotExtractor` - **Stream mode** (text-based, tuned for Microchip datasheets)
  - `PyMuPDFExtractor` - PyMuPDF TableFinder API
- `table_consensus.py` - Multi-extractor consensus with confidence scoring
  - `TableConsensusEngine` - Runs 3 extractors, computes agreement scores
  - `HybridConsensusEngine` - Adds optional LLM fallback
  - **NEW**: Separate header accuracy (30%) vs data row accuracy (70%) weighting
- `table_detection.py` - Pre-extraction validation (filters false positives)
- `table_evaluation.py` - Industry-standard metrics (TEDS, F1, STP rate)
- `table_multipage.py` - Multi-page continuation detection (Azure AI 2024 heuristics)
  - **Verified**: Successfully merges 114 rows across 3-page pinout tables
- `table_merging.py` - Vertical/horizontal table merging strategies
- `table_screenshots.py` - PyMuPDF-based screenshot generation for verification
- `table_validation.py` - Post-extraction validation (header detection, quality checks)
- `table_llm.py` - Vision LLM table extraction (fallback for low-confidence cases)

**Data Flow**:
```
PDF → [Extractor 1, Extractor 2, Extractor 3] → TableMatch → ConsensusResult
                                                     ↓
                                   Confidence Score (0.0-1.0)
                                   ├─ 40% Agreement (extractors present)
                                   ├─ 30% Structure (row/col consistency)
                                   └─ 30% Cell Similarity
                                        ├─ 30% Header Accuracy
                                        └─ 70% Data Row Accuracy (critical!)
```

**Key Design Patterns**:
- All extractors return `ExtractionResult` dataclass
- Tables represented as `ExtractedTable` with metadata (confidence, bbox, complexity)
- Consensus engine computes weighted scores with separate header/data accuracy
- **Camelot Mode Selection**: Stream mode (text-based) outperforms lattice (line-based) on Microchip datasheets (77 vs 55 rows captured)

**Recent Improvements (2025-10-17)**:
- Fixed CamelotExtractor parameter handling (lattice vs stream mode separation)
- Switched default from lattice to stream mode (40% more rows captured)
- Added header vs data row accuracy metrics to distinguish cosmetic vs critical differences
- Verified multi-page detection on real 3-page pinout tables (pages 30-32, 114 rows)
- See `docs/CAMELOT_TUNING.md` for detailed tuning results

---

### 2. Storage (`mchp_mcp_core/storage/`)

**Purpose**: Persist and retrieve data (vector embeddings + metadata).

**Modules**:
- `vector_store.py` - Qdrant vector store with hybrid search (BM25 + vector RRF)
- `schemas.py` - Pydantic schemas for DocumentChunk, SearchQuery, SearchResult
- *(ChromaDB and SQLite cache modules exist but not actively used)*

**Data Flow**:
```
DocumentChunk → Embeddings → Qdrant (vector + sparse BM25) → Hybrid Search
```

**Key Features**:
- Hybrid search: Combines dense vectors + BM25 sparse vectors with RRF
- Deduplication via SHA-256 content hashing
- Metadata filtering (product family, version, document type)

---

### 3. Embeddings (`mchp_mcp_core/embeddings/`)

**Purpose**: Generate vector embeddings for semantic search.

**Modules**:
- `embeddings.py` - sentence-transformers wrapper with caching

**Features**:
- Auto device detection (CPU/CUDA/MPS)
- Batch processing with progress bars
- Optional disk caching

**Typical Model**: `BAAI/bge-small-en-v1.5` (384 dimensions)

---

### 4. LLM (`mchp_mcp_core/llm/`)

**Purpose**: HTTP client for LLM API calls with retry logic and rate limiting.

**Modules**:
- `client.py` - Async HTTP client with tenacity retry

**Features**:
- Exponential backoff retry (3 attempts)
- Rate limiting
- PII redaction before sending
- Streaming support

---

### 5. Security (`mchp_mcp_core/security/`)

**Purpose**: PII redaction and path validation.

**Modules**:
- `pii.py` - Regex-based PII redaction (emails, phones, SSNs, credit cards)
- `path_validation.py` - Safe path handling and filename sanitization

---

### 6. Utils (`mchp_mcp_core/utils/`)

**Purpose**: Shared utilities and configuration.

**Modules**:
- `config.py` - Pydantic Settings-based configuration with secret masking
- `logging.py` - Structured logging with secret redaction
- `models.py` - `ExtractedChunk` dataclass for extraction pipeline

---

## Data Structures

### Core Types

```python
@dataclass
class ExtractedTable:
    """Represents an extracted table."""
    data: List[List[str]]
    page_num: int
    table_index: int = 0
    confidence: float = 0.0
    bbox: Optional[Tuple[float, float, float, float]] = None
    complexity: TableComplexity = TableComplexity.SIMPLE

@dataclass
class TableMatch:
    """Multi-extractor consensus match."""
    table_index: int
    page_num: int
    versions: Dict[str, ExtractedTable]  # extractor_name -> table
    agreement_score: float = 0.0
    cell_similarity: float = 0.0
    structure_score: float = 0.0
    confidence: float = 0.0  # Weighted: 40% agreement + 30% structure + 30% cell
    best_version: Optional[ExtractedTable] = None

@dataclass
class TableSpan:
    """Multi-page table span."""
    start_page: int
    end_page: int
    table_indices: List[int]
    continuation_type: ContinuationType  # VERTICAL or HORIZONTAL
    confidence: float = 0.0
    indicators: List[str] = field(default_factory=list)
```

---

## Key Workflows

### 1. Table Extraction with Consensus

```python
from mchp_mcp_core.extractors import TableConsensusEngine

engine = TableConsensusEngine(
    extractors=["pdfplumber", "camelot", "pymupdf"],
    enable_detection_filter=True
)

result = engine.extract_with_consensus("doc.pdf", page_num=5)

for match in result.matches:
    if match.confidence >= 0.8:
        table = match.best_version  # Highest quality version
```

### 2. Multi-Page Table Detection

```python
results = engine.extract_with_multipage_detection(
    pdf_path="datasheet.pdf",
    page_range=(21, 22),
    extractor_name="pdfplumber"
)

# Tables spanning multiple pages are merged automatically
merged_table = results[21].matches[0].best_version  # Contains merged data
```

### 3. Manual Verification with Screenshots

```python
from mchp_mcp_core.extractors import TableScreenshotGenerator

gen = TableScreenshotGenerator(dpi=300)

screenshots = gen.capture_table_with_versions(
    pdf_path="doc.pdf",
    page_num=5,
    extractor_tables={"pdfplumber": table1, "camelot": table2},
    output_dir="screenshots"
)
# Returns: {"pdfplumber": "screenshot1.png", "camelot": "screenshot2.png"}
```

---

## Performance Characteristics

### Table Extraction
- **Consensus extraction**: ~3-6 seconds per page (3 extractors in parallel)
- **Multi-page detection**: +1-2 seconds per page pair
- **Screenshot generation**: ~0.5 seconds per table (300 DPI)

### Embeddings
- **BGE-small**: ~200 chunks/second (CPU), ~1000 chunks/second (GPU)
- **Model loading**: ~2 seconds (cached after first load)

### Vector Search
- **Hybrid search**: <50ms for 10K chunks (Qdrant in-memory)
- **BM25 + vector RRF**: ~2x slower than pure vector, but better quality

---

## Current Limitations

### Table Extraction (Phase 3A Findings)

**Multi-page detection depends on extractor quality**:
- Detection requires column structure consistency (>80% similarity)
- Complex layouts (merged cells, spanning headers) confuse extractors
- Different extractors produce different column counts for same table
- Example: PIC32CZ pages 21-22
  - pdfplumber: 22 cols (page 21) → 9 cols (page 22) ❌
  - Continuation not detected due to mismatch

**Implications**:
- Automatic detection works well for simple tables
- Complex datasheets require manual verification
- Phase 3B (manual UI) is critical path forward

---

## Planned Enhancements

### Phase 3B: Manual Verification UI (NEXT)
- Screenshot-based table review interface
- Side-by-side comparison of 3 extractor versions
- Ground truth JSON export for benchmarking

### Phase 3C: Benchmarking
- Validate quality metrics on real datasheets
- Measure STP (Straight Through Processing) rate
- Document per-extractor accuracy

### Future
- Relaxed multi-page detection mode (fuzzy column matching)
- Hybrid detection (aggregate signals from multiple extractors)
- LLM-assisted continuation detection
- Improved extractor preprocessing (merged cell detection)

---

## Testing Strategy

### Test Organization
```
tests/
├── smoke_test.py                    # Quick sanity check
├── test_multipage_detection.py     # Multi-page detection tests
├── check_extractors.py             # Diagnostic inspection
└── table_extraction/               # Domain-specific suites
```

### Test Data
- `test_datasheets/` - Real Microchip datasheets for integration testing
- Focus on challenging cases (multi-page tables, complex layouts)

---

## Dependencies

**Core**:
- PyMuPDF (`fitz`) - PDF parsing and rendering
- pdfplumber - Text-based table extraction
- camelot-py[cv] - Line-based table extraction
- sentence-transformers - Embedding models
- qdrant-client - Vector database

**Optional**:
- openai - Vision LLM fallback (disabled by default)
- anthropic - Claude vision models (disabled by default)

**Infrastructure**:
- Pydantic - Configuration and validation
- tenacity - Retry logic
- httpx - Async HTTP client

---

## Configuration

Configuration via Pydantic Settings with environment variable support:

```python
from mchp_mcp_core.utils import BaseConfig

class MyConfig(BaseConfig):
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "microchip_docs"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    class Config:
        env_prefix = "MCHP_"
```

Environment variables: `MCHP_QDRANT_URL`, `MCHP_COLLECTION_NAME`, etc.

---

## Maintenance Notes

### When Adding New Extractors
1. Implement `TableExtractor` interface
2. Add to `table_extractors.py`
3. Register in `TableConsensusEngine.extractors_map`
4. Update `__init__.py` exports
5. **Update this file** with new extractor details

### When Changing Data Structures
1. Update all affected modules
2. Update serialization/deserialization code
3. Update schema validation
4. **Update this file** with new structures

### When Refactoring
1. Maintain backward compatibility where possible
2. Update all import statements
3. Update tests
4. **Update this file** with new organization

---

**Remember**: This file is the source of truth for architecture. Keep it current!
