# Source Projects Map

Comprehensive map of all projects contributing to mchp-mcp-core.

## 📂 Project Locations

### Active Projects

| Project | Path | Purpose | Key Components Extracted |
|---------|------|---------|-------------------------|
| **mchp-mcp-core** | `/home/jorgill/mchp-mcp-core` | Core library (this repo) | All modules |
| **mchp-MiNi-fpga-search** | `/home/jorgill/mchp-MiNi-fpga-search` | FPGA RAG system (production) | Qdrant, schemas, metadata extraction, PPTX |
| **datasheet-review** | `/home/jorgill/datasheet-review` | Datasheet review automation | ChromaDB, SQLite state, PDF extraction, validation |
| **fpga_mcp** | `/home/jorgill/fpga_mcp` | FPGA MCP server | Manifest system, hashing |
| **mchp-socket-intel-mcp** | `/home/jorgill/mchp-socket-intel-mcp` | Market intelligence scraping | Web scrapers (not extracted yet) |
| **microchip-fae-prompt-library** | `/home/jorgill/microchip-fae-prompt-library` | Prompt templates | (Domain-specific, not extracted) |

## 🗺️ Extraction History (Phase 1 & 2A)

### Phase 1: Foundation + Core Features

#### From mchp-MiNi-fpga-search

**Extracted Modules:**
- ✅ `storage/qdrant.py` (554 LOC) - Production Qdrant wrapper with hybrid search
- ✅ `storage/schemas.py` (218 LOC) - Pydantic data models
- ✅ `embeddings/sentence_transformers.py` (218 LOC) - Embedding wrapper
- ✅ `extractors/pptx.py` (131 LOC) - PPTX slide extraction
- ✅ `security/pii.py` (200 LOC) - PII redaction
- ✅ `security/validation.py` (138 LOC) - Path validation
- ✅ `utils/logger.py` (132 LOC) - Structured logging
- ✅ `utils/config.py` (192 LOC) - Pydantic Settings

**Still Available in MiNi:**
- `ingest/ingest_local.py` - Ingestion orchestration (may extract in Phase 2E)
- `ingest/metadata_extractor.py` - Metadata extraction (HIGH PRIORITY - Phase 2B)
- `serve/api.py` - FastAPI server (domain-specific)
- `serve/app.py` - UI app (domain-specific)
- `tests/batch_eval.py` - Evaluation framework (may extract)

#### From datasheet-review

**Extracted Modules:**
- ✅ `extractors/pdf.py` (311 LOC) - PDF extraction with structure preservation
- ✅ `extractors/tables.py` (255 LOC) - Multi-strategy table extraction
- ✅ `extractors/chunking.py` (310 LOC) - Fixed & semantic chunking
- ✅ `storage/chromadb.py` (276 LOC - Phase 2A) - ChromaDB wrapper
- ✅ `storage/sqlite.py` (350 LOC - Phase 2A) - SQLite caching
- ✅ `llm/client.py` (226 LOC) - Async LLM client

**Still Available in datasheet-review:**
- `src/database.py` - Review state management (partially extracted as sqlite.py)
- `src/terminology_analyzer.py` - Terminology consistency checking (Phase 2C)
- `src/completeness_validator.py` - Quality validation (Phase 2C)
- `src/output.py` - Markdown generation (Phase 2D)
- `src/pattern_library.py` - Validation patterns (Phase 2C)
- `src/review_*.py` - Domain-specific review modules

#### From fpga_mcp

**Extracted Modules:**
- ✅ `storage/manifest.py` (218 LOC - Phase 2A) - Document versioning system

**Still Available in fpga_mcp:**
- `src/fpga_rag/utils/hashing.py` - File hashing (Phase 2B - HIGH PRIORITY)
- `src/fpga_rag/ingestion/orchestrator.py` - Pipeline coordination (Phase 2E)
- `src/fpga_rag/utils/pdf.py` - Additional PDF utilities

## 📊 Extraction Status by Module

### ✅ Complete Modules (Production-Ready)

| Module | Files | Total LOC | Source Projects | Status |
|--------|-------|-----------|-----------------|--------|
| **extractors/** | 4 | 1,007 | datasheet-review, MiNi | ✅ Complete |
| **storage/** | 5 | 1,619 | MiNi, datasheet-review, fpga_mcp | ✅ Complete (Phase 2A) |
| **embeddings/** | 1 | 231 | MiNi | ✅ Complete |
| **llm/** | 1 | 239 | datasheet-review | ✅ Complete |
| **security/** | 2 | 364 | MiNi | ✅ Complete |
| **utils/** | 2 | 358 | MiNi | ✅ Complete |
| **models/** | 1 | 119 | MiNi | ✅ Complete |

**Total: 16 files, ~3,937 LOC**

### ⏳ Planned Extractions (Phase 2B-E)

| Priority | Module | Source | Estimated LOC | Phase |
|----------|--------|--------|---------------|-------|
| 🔴 HIGH | `extractors/metadata.py` | MiNi | 313 | 2B |
| 🔴 HIGH | `utils/hashing.py` | fpga_mcp | 80 | 2B |
| 🟡 MEDIUM | `analysis/terminology.py` | datasheet-review | 400 | 2C |
| 🟡 MEDIUM | `validation/quality.py` | datasheet-review | 300 | 2C |
| 🟡 MEDIUM | `output/markdown.py` | datasheet-review | 200 | 2D |
| 🟢 LOW | `ingestion/orchestrator.py` | MiNi, fpga_mcp | 300 | 2E |

### ❌ Not Extracting (Domain-Specific)

- mchp-socket-intel-mcp scrapers (market intelligence specific)
- microchip-fae-prompt-library (prompt templates specific)
- MiNi serve/ modules (FPGA RAG UI specific)
- datasheet-review domain logic (review-specific)

## 🛠️ Tool Selection Intelligence

### Vector Stores: Qdrant vs ChromaDB

**Use Qdrant when:**
- ✅ Production deployment
- ✅ Need hybrid search (BM25 + vector)
- ✅ Large-scale (>100K documents)
- ✅ Advanced filtering required
- ✅ Performance critical

**Use ChromaDB when:**
- ✅ Development/prototyping
- ✅ Small to medium datasets (<50K documents)
- ✅ Simple setup (no server)
- ✅ Local experimentation

**Evidence:**
- MiNi uses Qdrant: 10K+ FPGA documents, 80% P@3, 39ms latency
- datasheet-review uses ChromaDB: single-document review workflow

### Embedding Models

**Current Standard: `BAAI/bge-small-en-v1.5`**
- ✅ 384 dimensions (fast, compact)
- ✅ Excellent for technical documents
- ✅ Auto device detection (CPU/CUDA/MPS)
- ✅ 32 batch size optimal

**When to use alternatives:**
- Multilingual: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Higher quality: `BAAI/bge-base-en-v1.5` (768 dim, slower)
- Domain-specific: Fine-tune on your data

### Chunking Strategies

**Fixed-size (chunk_size=1500, overlap=200):**
- ✅ Default choice
- ✅ Predictable performance
- ✅ Works for 90% of use cases
- Evidence: Used in 3/5 projects

**Semantic chunking:**
- ✅ Better context preservation
- ✅ Section-aware
- ⚠️ More complex, slower
- Use when: Section hierarchy is critical

### LLM Integration Patterns

**Async + Retry (Recommended):**
- ✅ Tenacity for exponential backoff
- ✅ PII redaction before API calls
- ✅ Rate limiting with semaphore
- Evidence: Prevents cascading failures

**Streaming:**
- Only when: Real-time UX required
- Trade-off: More complex error handling

### State Management

**SQLite for:**
- ✅ Caching (exact retrieval fallback)
- ✅ Manifest/version tracking
- ✅ Review state
- ✅ Embeddings fallback when vector store down

**Not for:**
- ❌ Primary vector storage (use Qdrant/ChromaDB)
- ❌ High-concurrency writes

## 📈 Growth Trajectory

### Current (v0.1.0 - Phase 1 + 2A Complete)
- 16 files
- ~3,937 LOC
- 3 storage backends (Qdrant, ChromaDB, SQLite)
- 5 projects consolidated

### Near-term (v0.2.0 - Phase 2B-D)
- +4 files
- ~5,230 LOC
- Metadata extraction, hashing, analysis, validation, output generation

### Long-term (v0.3.0+)
- Ingestion orchestration
- More extractors (DOCX, HTML)
- Reranking support
- Fine-tuning utilities

## 🔗 Project Dependencies

```
mchp-mcp-core (foundation)
    ↑
    ├── mchp-MiNi-fpga-search (will migrate to use core)
    ├── datasheet-review (will migrate to use core)
    ├── fpga_mcp (will migrate to use core)
    └── Future MCP servers (thin wrappers on core)
```

## 📝 Maintenance Notes

**Before extracting a module, check:**
1. Is it used by 2+ projects? (2+ rule)
2. Is it domain-agnostic? (no FPGA-specific logic)
3. Does it have clean interfaces? (no tight coupling)
4. Is it tested? (or can we write tests)

**Extraction checklist:**
- [ ] Read source module
- [ ] Identify dependencies
- [ ] Refactor to use core interfaces (Pydantic, logger, config)
- [ ] Remove domain-specific logic
- [ ] Add type hints
- [ ] Update __init__.py exports
- [ ] Add to smoke tests
- [ ] Update README
- [ ] Document in PROJECTS_MAP.md
