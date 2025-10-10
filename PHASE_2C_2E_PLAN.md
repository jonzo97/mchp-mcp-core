# Phase 2C-E Development & Tech Stack Research Plan

**Status**: Phase 2B Complete → Phase 2C In Progress
**Date**: 2025-10-09
**Current Version**: v0.1.2 (18 modules, 4,678 LOC)

---

## Executive Summary

This document outlines the development roadmap for completing Phases 2C-2E of mchp-mcp-core library development, plus comprehensive tech stack research to identify blind spots and optimization opportunities for our target applications.

**Selected Approach**: **Option E (Hybrid Strategy)**
- Execute Phase 2C (Analysis & Validation modules) for immediate value
- Create tech stack research documentation in parallel
- Review gaps and plan subsequent phases

---

## Current State Assessment

### ✅ Completed Phases

**Phase 1 - Foundation** (v0.1.0)
- Extractors: PDF, PPTX, tables, chunking (4 files, 1,007 LOC)
- Storage: Qdrant vector store with schemas (2 files, 772 LOC)
- Embeddings: sentence-transformers wrapper (1 file, 231 LOC)
- LLM: Async client with retry (1 file, 239 LOC)
- Security: PII redaction, validation (2 files, 364 LOC)
- Utils: Config, logging (2 files, 358 LOC)
- Models: Shared schemas (1 file, 119 LOC)

**Phase 2A - Storage Completeness** (v0.1.1)
- ChromaDB: Development-friendly vector store (276 LOC)
- SQLite: Async caching layer (350 LOC)
- Manifest: Document versioning system (218 LOC)

**Phase 2B - Metadata & Hashing** (v0.1.2)
- Metadata Extraction: Document type, product, version, date (372 LOC)
- File Hashing: Checksums and deduplication (149 LOC)

### 🎯 Library Metrics (v0.1.2)

- **Total Files**: 18 modules
- **Total LOC**: 4,678
- **Storage Backends**: 3 (Qdrant, ChromaDB, SQLite)
- **Dependencies**: 12 core + 2 async + 6 dev
- **Test Coverage**: 100% smoke tests passing
- **Git Status**: All commits pushed to origin/master

---

## Phase 2C: Analysis & Validation Modules

**Priority**: HIGH
**Estimated Effort**: 3-4 hours
**Target Version**: v0.1.3

### Module 1: `analysis/terminology.py`

**Source**: `datasheet-review/src/terminology_analyzer.py` (404 LOC)

**Purpose**: Detect terminology inconsistencies and variations across documentation

**Core Features**:
- Technical term extraction with regex patterns
- Canonical form normalization (lowercase, no special chars)
- Variation detection (Wi-Fi vs WiFi, I²C vs I2C)
- Category-based grouping (connectivity, peripherals, features, memory)
- Severity scoring (critical, high, medium, low)
- Location tracking (chunk_id, page_number)

**Key Classes**:
```python
@dataclass
class Term:
    term_id: str
    canonical_form: str
    actual_form: str
    category: str
    locations: List[Tuple[str, int]]
    count: int

@dataclass
class TermVariation:
    canonical: str
    variations: Dict[str, int]
    category: str
    total_instances: int
    recommended_form: str
    severity: str
    reason: str

class TerminologyAnalyzer:
    def __init__(self, config: Dict)
    def analyze_chunks(self, chunks: List) -> Dict
    def detect_variations(self, terms: List[Term]) -> List[TermVariation]
    def generate_report(self) -> Dict
```

**Refactoring Needed**:
- ✅ Use mchp_mcp_core.utils.logger instead of stdlib logging
- ✅ Make term patterns configurable (extend for domains beyond datasheets)
- ✅ Type hints with modern syntax (List → list, Dict → dict)
- ✅ Remove domain-specific patterns (keep generic, provide examples)
- ✅ Add batch processing support

**Dependencies**: None (uses stdlib: re, dataclasses, collections)

---

### Module 2: `validation/quality.py`

**Source**: `datasheet-review/src/completeness_validator.py` (550 LOC)

**Purpose**: Semantic completeness validation - ensure claims have supporting evidence

**Core Features**:
- LLM-based claim extraction from features/intro sections
- Evidence search (specs, tables, figures, sections)
- Relevance scoring between claims and evidence
- Missing evidence detection
- Validation report generation with severity levels

**Key Classes**:
```python
@dataclass
class Claim:
    claim_id: str
    claim_text: str
    claim_type: str  # 'feature', 'specification', 'capability'
    section: str
    page_number: int
    requires_evidence: List[str]
    confidence: float

@dataclass
class Evidence:
    evidence_id: str
    evidence_type: str  # 'section', 'table', 'figure', 'specification'
    content: str
    location: str
    relevance_score: float

@dataclass
class ValidationResult:
    claim: Claim
    evidence_found: List[Evidence]
    is_supported: bool
    confidence: float
    severity: str
    suggestion: str
    missing_evidence: List[str]

class CompletenessValidator:
    def __init__(self, config: Dict, llm_client, knowledge_base)
    async def validate_completeness(self, chunks: List, structure: Dict) -> Dict
    async def extract_claims(self, chunks: List) -> List[Claim]
    async def find_evidence(self, claim: Claim, chunks: List) -> List[Evidence]
    async def validate_claim(self, claim: Claim, evidence: List[Evidence]) -> ValidationResult
```

**Refactoring Needed**:
- ✅ Use mchp_mcp_core.llm.LLMClient instead of custom client
- ✅ Optional integration with mchp_mcp_core.storage for knowledge base
- ✅ Type hints modernization
- ✅ Make claim/evidence patterns configurable
- ✅ Async/await throughout

**Dependencies**:
- Uses existing: mchp_mcp_core.llm.LLMClient
- Optional: mchp_mcp_core.storage.QdrantVectorStore (for knowledge base)

---

## Phase 2D: Output Generation Modules

**Priority**: MEDIUM
**Estimated Effort**: 2 hours
**Target Version**: v0.1.4

### Module 1: `output/markdown.py`

**Source**: `datasheet-review/src/output.py` (297 LOC)

**Purpose**: Reassemble reviewed/processed chunks into final markdown documents

**Core Features**:
- Chunk sorting by page/position
- Document header generation with metadata
- Table of contents generation
- Section hierarchy preservation
- Change tracking annotations
- Cross-reference report integration
- Page reference comments

**Key Classes**:
```python
class MarkdownGenerator:
    def __init__(self, config: Dict)
    def generate_document(self, chunks: List, metadata: Dict,
                         cross_ref_report: Optional[Dict]) -> str
    def _generate_header(self, metadata: Dict) -> str
    def _generate_toc(self, chunks: List) -> str
    def _format_chunk(self, chunk: Dict) -> str
```

**Refactoring Needed**:
- ✅ Use mchp_mcp_core.models.DocumentChunk
- ✅ Separate concerns: formatting vs file I/O
- ✅ Make output templates configurable
- ✅ Support multiple output formats (prep for future HTML, DOCX)

### Module 2: `output/filter.py` (Optional)

**Source**: `datasheet-review/src/output_filter.py` (287 LOC)

**Purpose**: Content filtering and sanitization before output

---

## Phase 2E: Orchestration Module

**Priority**: LOW
**Estimated Effort**: 4 hours
**Target Version**: v0.1.5

### Module: `ingestion/orchestrator.py`

**Sources**:
- `mchp-MiNi-fpga-search/ingest/ingest_local.py` (batch processing patterns)
- `fpga_mcp/src/fpga_rag/ingestion/orchestrator.py` (manifest-based staging)

**Purpose**: Unified ingestion orchestration with job scheduling and progress tracking

**Core Features**:
- Directory scanning for documents
- Manifest-based deduplication (checksum)
- Job queue management
- Progress tracking with Rich tables
- Error handling and retry logic
- Batch processing with concurrency control
- Report generation

**Key Classes**:
```python
@dataclass
class IngestionJob:
    path: Path
    manifest: DocumentManifest
    priority: int
    retry_count: int

class IngestionOrchestrator:
    def __init__(self, config: Config, manifest_repo: ManifestRepository)
    def stage_from_directory(self, directory: Path) -> List[IngestionJob]
    async def process_job(self, job: IngestionJob) -> IngestionResult
    async def run_batch(self, jobs: List[IngestionJob]) -> BatchReport
    def generate_report(self, results: List[IngestionResult]) -> str
```

**Refactoring Needed**:
- ✅ Integrate with mchp_mcp_core.storage.ManifestRepository
- ✅ Use mchp_mcp_core.extractors for document processing
- ✅ Use mchp_mcp_core.embeddings.EmbeddingModel
- ✅ Use mchp_mcp_core.storage vector stores
- ✅ Async processing with asyncio
- ✅ Rich progress bars and tables

---

## Tech Stack Research Plan

**Deliverable**: `TECH_STACK.md` - Comprehensive technology landscape analysis

### Research Areas

#### 1. Reranking Options (Currently Missing)

**Current State**:
- MiNi achieved P@3=80% without reranking
- No reranking implementation in any project
- Qdrant native RRF (BM25 + vector) provides first-stage ranking

**Research Questions**:
- Would reranking improve P@1 score?
- What's the latency trade-off?
- Which reranking approach for technical docs?

**Options to Evaluate**:

| Approach | Model | Pros | Cons | Use Case |
|----------|-------|------|------|----------|
| Cross-Encoder | sentence-transformers/ms-marco-MiniLM-L-12-v2 | High accuracy, OSS | Slow (pairwise), 512 token limit | Top-20 → Top-5 |
| Cross-Encoder | BAAI/bge-reranker-base | Technical docs optimized | Requires GPU for speed | Technical documentation |
| API-based | Cohere Rerank | Fast, no hosting | Cost, external dependency | Production with budget |
| ColBERT | stanford-nlp/colbert-v2 | Fast late interaction | Complex setup, resource intensive | Large-scale systems |
| Learned Sparse | SPLADE | Interpretable, fast | Less accurate than cross-encoder | Hybrid with BM25 |

**Benchmark Plan**:
- Use MiNi's test queries (tests/polarfire_queries.csv)
- Measure P@1, P@3, P@5 with/without reranking
- Measure latency impact
- Cost analysis (if using API)

**Recommendation Criteria**:
- Deploy reranker if P@1 < 40% and reranking improves by >10%
- Latency increase < 200ms acceptable
- Prefer OSS for internal deployment

---

#### 2. Storage Backend Comparison

**Current State**: 3 backends implemented
- Qdrant (production, hybrid search)
- ChromaDB (development, no server)
- SQLite (caching, exact retrieval)

**Additional Options to Research**:

| Backend | Type | Hybrid Search | Filtering | Pros | Cons | Use Case |
|---------|------|---------------|-----------|------|------|----------|
| **Weaviate** | Vector DB | ✅ BM25+vector | Advanced | GraphQL, multi-tenancy | Complex setup | Enterprise, ACL |
| **Milvus** | Vector DB | ✅ (plugin) | Good | Scalable, Kubernetes | Heavy, complex | Large-scale |
| **Pinecone** | Managed | ❌ (vector only) | Basic | Managed, easy | Cost, vendor lock | Quick MVP |
| **Postgres+pgvector** | SQL+Vector | ❌ (separate queries) | Excellent | Familiar, transactions | Not optimized for vectors | Existing Postgres |
| **Elasticsearch** | Search | ✅ Native | Excellent | Mature, analytics | Resource intensive | Search-heavy apps |
| **Redis** | In-memory | ❌ | Basic | Ultra-fast, familiar | Volatile, limited features | Caching layer |

**Evaluation Matrix**:
- Hybrid search quality (BM25+vector fusion)
- Filtering performance (product_family, date range, etc.)
- Scalability (100K+ documents)
- Ease of deployment (Docker, cloud, serverless)
- Cost (hosting, licensing)
- Community and support

**Recommendations**:
- Keep Qdrant as primary (proven, OSS, hybrid search)
- Keep ChromaDB for development
- Consider Weaviate if ACL/multi-tenancy needed
- Consider Postgres+pgvector if already using Postgres

---

#### 3. Embedding Model Benchmarking

**Current State**:
- bge-small-en-v1.5 (384 dim) - fast, compact
- Proven on 10K+ FPGA docs (80% P@3)

**Alternative Models to Benchmark**:

| Model | Dimensions | Size | Pros | Cons | Use Case |
|-------|------------|------|------|------|----------|
| **bge-small-en-v1.5** ✅ | 384 | 33MB | Current, proven | Lower quality vs large | Production (current) |
| **bge-base-en-v1.5** | 768 | 109MB | Higher quality | 2x slower, 2x storage | Quality-critical |
| **bge-large-en-v1.5** | 1024 | 326MB | Highest quality | 4x slower, 3x storage | Offline processing |
| **OpenAI ada-002** | 1536 | API | Good quality | Cost, external API | Rapid prototyping |
| **Cohere embed-v3** | 1024 | API | Latest, multilingual | Cost, external API | Multilingual docs |
| **E5-base-v2** | 768 | 109MB | Instruction-based | Training data quality? | Task-specific prompts |
| **Fine-tuned bge** | 384 | 33MB | Domain-optimized | Requires labeled data | Large training set |

**Benchmark Plan**:
1. Use MiNi's evaluation framework (tests/batch_eval.py)
2. Test queries: PolarFire queries (20 queries)
3. Metrics: P@1, P@3, P@5, MRR, latency
4. Embedding time: Single doc, batch (100 docs)
5. Storage impact: Index size for 10K docs

**Fine-Tuning Evaluation**:
- Requires: 500+ query-document pairs
- Approach: Sentence-transformers fine-tuning on contrastive pairs
- Expected improvement: +5-15% on P@3
- Trade-off: Maintenance burden (retrain on new domains)

**Recommendation Criteria**:
- Stick with bge-small if P@3 > 75%
- Upgrade to bge-base if P@3 < 70% and latency acceptable
- Fine-tune if have 500+ labeled pairs and P@3 < 70%

---

#### 4. Advanced NLP Pipeline Options

**Current State**: Basic text extraction, no NLP processing

**Potential Enhancements**:

**A. Named Entity Recognition (NER)**
- Extract IC part numbers, protocols, standards
- Tools: spaCy, Hugging Face NER models
- Use case: Auto-tag documents with detected entities
- Effort: Medium (need labeled data for custom entities)

**B. Dependency Parsing**
- Extract technical relationships (X supports Y, requires Z)
- Tools: spaCy dependency parser
- Use case: Knowledge graph construction
- Effort: High (requires domain expertise)

**C. Abbreviation Expansion**
- Detect and expand technical abbreviations
- Tools: scispacy, custom dictionaries
- Use case: Improve search recall (SPI → Serial Peripheral Interface)
- Effort: Low (dictionary-based)

**D. Section Classification**
- Auto-classify sections (Features, Specs, Pinout, etc.)
- Tools: Text classification models (BERT-based)
- Use case: Better chunk metadata, targeted search
- Effort: Medium (need training data)

**Recommendation**:
- Phase 1: Abbreviation expansion (low-hanging fruit)
- Phase 2: Section classification (improves metadata)
- Phase 3: NER for domain entities (if knowledge graph needed)
- Skip dependency parsing (high effort, unclear ROI)

---

#### 5. Document Processing Enhancements

**Current State**:
- PDF: PyMuPDF (text + structure)
- PPTX: python-pptx (slide text)
- Tables: pdfplumber (multi-strategy)

**Enhancement Options**:

**A. OCR for Scanned PDFs**
- Current: Text-based PDFs only
- Tools: Tesseract (OSS), AWS Textract (managed)
- Trade-offs: Accuracy vs cost vs latency
- Recommendation: Tesseract for infrequent scans, Textract if high volume

**B. Layout Analysis**
- Detect document structure (headers, footers, columns)
- Tools: unstructured.io, layout-parser, PDFMiner
- Use case: Better chunk boundaries, preserve reading order
- Recommendation: Evaluate if current chunking inadequate

**C. DOCX Extraction**
- Current: Not supported
- Tools: python-docx, docx2txt
- Use case: Word documents (common in corporate settings)
- Recommendation: High priority if target corpus includes DOCX

**D. HTML/Markdown Extraction**
- Current: Not supported
- Tools: BeautifulSoup, trafilatura, html2text
- Use case: Web documentation, Markdown READMEs
- Recommendation: Medium priority (depends on corpus)

**E. Image/Figure Extraction**
- Current: Basic figure detection in PDFs
- Enhancement: Extract, caption, describe with vision models
- Tools: PyMuPDF (extraction) + CLIP/BLIP (description)
- Use case: Search across figures, answer visual questions
- Recommendation: Future (Phase 3+)

**Priority Ranking**:
1. DOCX extraction (high demand, low effort)
2. OCR for scanned PDFs (common need, moderate effort)
3. HTML extraction (useful for web docs, low effort)
4. Layout analysis (only if chunking quality issues)
5. Image description (future, high effort)

---

#### 6. Async Architecture Patterns

**Current State**:
- LLM client: httpx async + tenacity retry
- SQLite: aiosqlite
- Limited async orchestration

**Enhancement Areas**:

**A. Concurrent Document Processing**
- Pattern: asyncio.gather for batch extraction
- Benefit: 5-10x speedup on multi-doc ingestion
- Consideration: Rate limiting, resource constraints

**B. Streaming LLM Responses**
- Pattern: SSE (Server-Sent Events) for UI
- Benefit: Better UX for long responses
- Consideration: Complexity in error handling

**C. Task Queue Integration**
- Tools: Celery, RQ, Dramatiq, Temporal
- Use case: Long-running ingestion jobs
- Benefit: Decoupled, scalable processing
- Recommendation: If processing > 1000 docs/day

**D. Rate Limiting Strategies**
- Current: Basic tenacity retry
- Enhancement: Token bucket, sliding window
- Tools: aiolimiter, pyrate-limiter
- Use case: API cost control, server protection

**E. Connection Pooling**
- Current: Single httpx client instance
- Enhancement: Connection pool tuning
- Benefit: Better throughput for concurrent requests

**Recommendation**:
- Implement concurrent extraction immediately (low effort, high value)
- Add task queue if processing > 1000 docs/day
- Streaming LLM optional (UX enhancement)
- Rate limiting critical if using paid APIs

---

## Blind Spot Analysis

### Potential Gaps in Current Stack

1. **Observability** ❌
   - No metrics collection (latency, success rate)
   - No distributed tracing
   - Basic logging only
   - **Recommendation**: Add structlog, OpenTelemetry

2. **Configuration Management** ⚠️
   - Pydantic Settings (good for env vars)
   - No feature flags, A/B testing
   - **Recommendation**: Add feature flag system (if needed)

3. **Caching Strategy** ⚠️
   - Embedding caching (disk-based)
   - No HTTP response caching
   - **Recommendation**: Add Redis layer for API responses

4. **Error Recovery** ⚠️
   - Retry logic in LLM client
   - No circuit breaker pattern
   - **Recommendation**: Add circuit breaker (tenacity or pybreaker)

5. **Testing Coverage** ⚠️
   - Smoke tests only
   - No integration tests
   - No performance benchmarks
   - **Recommendation**: Phase 3 priority

6. **Security** ⚠️
   - PII redaction ✅
   - Path validation ✅
   - No input sanitization for LLM prompts
   - No rate limiting per user
   - **Recommendation**: Add prompt injection detection

7. **Scalability** ⚠️
   - No horizontal scaling story
   - No load balancing
   - Single-instance vector store
   - **Recommendation**: Kubernetes patterns if > 10 concurrent users

8. **Cost Optimization** ❌
   - No cost tracking for API calls
   - No query result caching
   - **Recommendation**: Add usage tracking module

---

## Target Application Fit Analysis

### Application 1: mchp-MiNi-fpga-search (FPGA RAG)
**Current Gaps**:
- ✅ Core features well-covered
- ⚠️ Missing: Batch evaluation framework (in MiNi but not extracted)
- ⚠️ Missing: Web UI components (Flask/FastAPI servers - domain-specific)

**Recommendations**:
- Extract batch_eval.py → evaluation/ module
- Keep UI layer in application (domain-specific)

### Application 2: datasheet-review (Document QA)
**Current Gaps**:
- ⚠️ Language tools not extracted (grammar, spell check)
- ⚠️ Pattern library not extracted (institutional learning)
- ⚠️ Knowledge base not extracted (RAG enhancement)
- ⚠️ Diff/comparison not extracted

**Recommendations**:
- Phase 2C covers terminology + validation ✅
- Consider pattern_library + knowledge_base (Phase 2F?)
- Diff tools (version comparison) - Phase 2G?

### Application 3: fpga_mcp (MCP Server)
**Current Gaps**:
- ✅ Core features covered (manifest, hashing, storage)
- ⚠️ Orchestration not extracted (Phase 2E)
- ⚠️ MCP server wrapper patterns not documented

**Recommendations**:
- Phase 2E covers orchestration ✅
- Create MCP integration guide (separate doc)

### Application 4: Future MCP Servers
**Potential Gaps**:
- Authentication/authorization patterns
- Multi-tenant isolation
- Webhook/callback patterns
- Streaming responses

**Recommendations**:
- Document MCP best practices
- Create reference implementation

---

## Success Metrics

### Code Quality
- ✅ 100% type hints
- ✅ Comprehensive docstrings
- ✅ Pydantic validation throughout
- ⬜ Unit test coverage > 80% (future)
- ⬜ Integration tests (future)

### Reusability
- ✅ 2+ projects use each module
- ✅ Domain-agnostic interfaces
- ✅ Configurable patterns
- ⬜ Published to internal PyPI (future)

### Performance
- ✅ Extraction: < 5s per document
- ✅ Search: < 50ms p95 latency
- ⬜ Ingestion: > 100 docs/hour (orchestration)

### Documentation
- ✅ README with examples
- ✅ PROJECTS_MAP with tool intelligence
- ✅ Inline docstrings
- ⬜ API reference docs (future)
- ⬜ Architecture decision records (future)

---

## Next Steps (Immediate)

### Phase 2C Execution (This Session)
1. ✅ Create planning document (this file)
2. ⏳ Extract terminology analyzer → `analysis/terminology.py`
3. ⏳ Extract completeness validator → `validation/quality.py`
4. ⏳ Create `TECH_STACK.md` research document
5. ⏳ Update exports and dependencies
6. ⏳ Run tests
7. ⏳ Update README examples
8. ⏳ Commit v0.1.3

### Post-Phase 2C Review
- Assess blind spots vs target applications
- Decide: Phase 2D (output) vs 2E (orchestration) vs 2F (pattern library)
- Prioritize tech stack enhancements based on real needs

---

## Timeline Estimates

| Phase | Deliverables | Effort | Target Version |
|-------|-------------|--------|----------------|
| **2C** ✅ | Terminology + Validation | 3-4h | v0.1.3 |
| **2D** | Output generation | 2h | v0.1.4 |
| **2E** | Orchestration | 4h | v0.1.5 |
| **2F** | Pattern library + KB | 3h | v0.1.6 |
| **2G** | Document comparison | 2h | v0.1.7 |
| **3A** | Reranking (if needed) | 4h | v0.2.0 |
| **3B** | Advanced NLP | 6h | v0.2.1 |
| **3C** | Evaluation framework | 3h | v0.2.2 |

**Total Phase 2**: ~14 hours remaining
**Phase 3**: ~13 hours (optional enhancements)

---

## Decision Log

### 2025-10-09: Selected Hybrid Approach (Option E)
**Decision**: Execute Phase 2C + create tech stack research doc in parallel

**Rationale**:
- Phase 2C modules proven in production (datasheet-review)
- High reusability across all target applications
- Tech stack research provides foundation for informed decisions
- Parallel execution maximizes session productivity

**Alternatives Considered**:
- Option A (2C only): Faster but doesn't address blind spots
- Option B (research only): No immediate code value
- Option C (evaluation framework): Important but lower priority than 2C

**Approval**: User confirmed "let's run E"

---

## References

- SESSION_SUMMARY.md - Phase 1-2B history
- PROJECTS_MAP.md - Source project map
- EXTRACTION_STATUS.md - Completeness verification
- README.md - Usage examples

---

**Next Update**: After Phase 2C completion
