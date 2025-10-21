# Technology Stack Research & Recommendations

**Date**: 2025-10-09
**Version**: 1.0
**Status**: Research Complete

This document provides comprehensive research on technology options for RAG systems, document processing, and MCP servers, with evidence-based recommendations for the mchp-mcp-core ecosystem.

---

## Executive Summary

### Current Stack (v0.1.3)
- **Storage**: Qdrant (production), ChromaDB (dev), SQLite (cache)
- **Embeddings**: bge-small-en-v1.5 (384 dim)
- **Extraction**: PyMuPDF, pdfplumber, python-pptx
- **LLM**: Async HTTP client with retry (httpx + tenacity)
- **Analysis**: Terminology consistency, semantic completeness validation

### Key Findings
✅ **No Critical Gaps**: Current stack covers all essential functionality
⚠️ **Reranking**: Not implemented, but MiNi achieved 80% P@3 without it
⚠️ **Observability**: No metrics/tracing (acceptable for MVP)
⚠️ **Testing**: Smoke tests only (functional coverage needed)

### Recommendations
1. **Keep current stack** - proven in production (MiNi)
2. **Add reranking** only if P@1 < 40% on new datasets
3. **Implement evaluation framework** before optimizations
4. **Add observability** when moving to production scale

---

## 1. Reranking Systems

### Current State
- MiNi: P@3=80% **without reranking** (Qdrant RRF only)
- No cross-encoder or reranking implementation in any project
- Qdrant native hybrid search (BM25 + vector RRF) provides first-stage ranking

### When to Add Reranking
**Deploy reranker if:**
- P@1 < 40% (first result quality matters)
- P@3 < 70% (top-3 results insufficient)
- Latency budget allows +100-300ms
- Have 100+ evaluation queries for benchmarking

### Reranking Options

| Approach | Model | Accuracy | Latency | Hosting | Use Case |
|----------|-------|----------|---------|---------|----------|
| **Cross-Encoder** | ms-marco-MiniLM-L-12-v2 | High | 100-300ms | Self-hosted | P@1 optimization |
| **Cross-Encoder** | BAAI/bge-reranker-base | Higher | 150-400ms | Self-hosted GPU | Technical docs |
| **API** | Cohere Rerank | High | 50-150ms | Cloud API | Quick MVP |
| **ColBERT** | colbert-v2 | Very High | 20-100ms | Complex setup | Large-scale |
| **Learned Sparse** | SPLADE | Medium | 10-50ms | Self-hosted | Hybrid fallback |

### Implementation Pattern

```python
# Pseudo-code for cross-encoder reranking
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        self.model = CrossEncoder('ms-marco-MiniLM-L-12-v2')

    def rerank(self, query: str, results: List[SearchResult], top_k: int = 5):
        # Only rerank top-20 from initial retrieval
        candidates = results[:20]

        # Score query-document pairs
        pairs = [(query, r.content) for r in candidates]
        scores = self.model.predict(pairs)

        # Rerank and return top-k
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [r for r, s in ranked[:top_k]]
```

### Benchmark Plan
1. Use MiNi's evaluation framework (tests/batch_eval.py)
2. Metrics: P@1, P@3, P@5, MRR, latency
3. Test on 20+ real user queries
4. Compare: No reranking vs cross-encoder vs API

### Recommendation
⏸️ **Defer reranking** until:
- Evaluation framework implemented (Phase 2C+)
- P@1 measured < 40% on production queries
- Business case for +$500/month inference cost (if API)

---

## 2. Vector Store Comparison

### Current Implementation
- **Qdrant**: Production (10K+ docs, 80% P@3, 39ms latency)
- **ChromaDB**: Development (no server, quick prototyping)
- **SQLite**: Caching (exact retrieval fallback)

### Alternative Options

| Backend | Hybrid Search | Filtering | Scalability | Deployment | Best For |
|---------|---------------|-----------|-------------|------------|----------|
| **Qdrant** ✅ | Native RRF | Excellent | 100M+ | Docker/Cloud | Production (current) |
| **ChromaDB** ✅ | Basic | Good | 10M | Embedded | Development (current) |
| **Weaviate** | BM25+vector | Advanced | 100M+ | K8s/Cloud | Enterprise ACL |
| **Milvus** | Plugin | Good | Billions | K8s | Massive scale |
| **Pinecone** | Vector only | Basic | Managed | API | Quick MVP |
| **Postgres+pgvector** | Separate | SQL | 100M | Standard | Existing Postgres |
| **Elasticsearch** | Native | Excellent | 100M+ | Complex | Search-heavy |

### Feature Comparison

**Qdrant Strengths:**
- Native hybrid search (no plugin needed)
- Advanced payload filtering
- Multi-vector support
- HNSW + quantization
- Docker single-command deploy
- OSS with optional cloud

**When to Consider Alternatives:**
- **Weaviate**: Need GraphQL API, multi-tenancy, complex ACLs
- **Postgres+pgvector**: Already using Postgres, need transactions
- **Elasticsearch**: Heavy analytics, existing ELK stack
- **Pinecone**: Rapid prototyping, no infrastructure management

### Recommendation
✅ **Keep Qdrant** - proven, OSS, excellent hybrid search
✅ **Keep ChromaDB** - development convenience
🔄 **Consider Weaviate** only if ACL requirements emerge

---

## 3. Embedding Models

### Current Model
- **bge-small-en-v1.5**: 384 dim, 33MB, fast
- **Performance**: 80% P@3 on 10K FPGA docs (MiNi)
- **Latency**: ~30ms per doc (CPU), ~5ms (GPU)

### Alternative Models

| Model | Dims | Size | Quality | Speed | Use Case |
|-------|------|------|---------|-------|----------|
| **bge-small-en-v1.5** ✅ | 384 | 33MB | Good | Fast | Current (proven) |
| **bge-base-en-v1.5** | 768 | 109MB | Better | 2x slower | Quality > speed |
| **bge-large-en-v1.5** | 1024 | 326MB | Best | 4x slower | Offline batch |
| **OpenAI ada-002** | 1536 | API | Very Good | API | Rapid proto |
| **Cohere embed-v3** | 1024 | API | Excellent | API | Multilingual |
| **E5-base-v2** | 768 | 109MB | Good | Medium | Instruction-based |

### Benchmark Plan
1. Test datasets: 1K FPGA docs + 20 eval queries
2. Metrics: P@1, P@3, P@5, MRR
3. Measure: Embedding time, index size, memory usage
4. Cost: API calls vs self-hosted GPU

### Expected Trade-offs

**bge-small → bge-base:**
- Quality: +5-10% on P@3
- Latency: 2x slower embedding
- Storage: 2x larger index
- Cost: 2x memory

**bge-small → OpenAI ada-002:**
- Quality: +10-15% on P@3 (est.)
- Latency: API call (~100ms)
- Cost: $0.0001/1K tokens (~$100/month for 10K docs)

### Fine-Tuning Evaluation

**Requirements:**
- 500+ labeled query-document pairs
- Training time: 2-4 hours on GPU
- Maintenance: Retrain per domain

**Expected Improvement:**
- +5-15% on domain-specific P@3
- Best for: Specialized corpora (legal, medical, highly technical)

**Recommendation for Fine-Tuning:**
❌ **Not recommended** unless:
- Have 500+ high-quality labels
- P@3 < 70% with bge-small
- Domain highly specialized (not general technical docs)

### Recommendation
✅ **Keep bge-small-en-v1.5** - proven, fast, good quality
🔄 **Upgrade to bge-base** if P@3 < 70% and latency budget allows
🔄 **Fine-tune** only with 500+ labels and P@3 < 70%

---

## 4. Advanced NLP Pipelines

### Current State
- Basic text extraction (PyMuPDF)
- No NER, dependency parsing, or linguistic analysis

### Potential Enhancements

#### A. Named Entity Recognition (NER)

**Use Case**: Extract part numbers, protocols, standards

**Options:**
- spaCy (general): en_core_web_sm (12MB), en_core_web_lg (742MB)
- Hugging Face: bert-base-NER, roberta-large-NER
- Custom: Fine-tune on IC part numbers, protocols

**Effort**: Medium (need labeled examples)

**Value**:
- ✅ Auto-tag documents with detected entities
- ✅ Improve search recall (search by part number)
- ⚠️ Requires training data for custom entities (SPI, I2C, etc.)

**Recommendation**: ⏸️ **Defer** - marginal value vs effort

---

#### B. Abbreviation Expansion

**Use Case**: Expand technical abbreviations for better search

**Implementation:**
```python
ABBREVIATIONS = {
    'SPI': 'Serial Peripheral Interface',
    'I2C': 'Inter-Integrated Circuit',
    'UART': 'Universal Asynchronous Receiver/Transmitter',
    # ...
}

def expand_abbreviations(text):
    for abbr, full in ABBREVIATIONS.items():
        text = text.replace(abbr, f"{abbr} ({full})")
    return text
```

**Effort**: Low (dictionary-based)

**Value**:
- ✅ Improves search recall ("SPI" matches "Serial Peripheral Interface")
- ✅ Better for non-expert users

**Recommendation**: ✅ **Implement** - low effort, clear value

---

#### C. Section Classification

**Use Case**: Auto-classify sections (Features, Specs, Pinout, etc.)

**Options:**
- Rule-based: Regex patterns on section headers
- ML-based: Text classification (BERT-based)

**Effort**: Low (rule-based), Medium (ML)

**Value**:
- ✅ Better chunk metadata
- ✅ Targeted search ("specs only", "features only")
- ✅ Evidence type classification (for validation module)

**Recommendation**: ✅ **Implement rule-based** - already partially done in metadata extractor

---

#### D. Dependency Parsing

**Use Case**: Extract relationships (X supports Y, requires Z)

**Effort**: High (requires linguistic expertise)

**Value**:
- ⚠️ Knowledge graph construction (complex)
- ⚠️ Unclear ROI for technical documentation

**Recommendation**: ❌ **Skip** - high effort, unclear value

---

### NLP Priority Ranking

1. ✅ **Abbreviation expansion** (Low effort, clear value)
2. ✅ **Section classification** (Low effort, improves metadata)
3. ⏸️ **NER for custom entities** (If have labeled data)
4. ❌ **Dependency parsing** (High effort, unclear ROI)

---

## 5. Document Processing Enhancements

### Current Capabilities
- ✅ PDF: Text + structure (PyMuPDF)
- ✅ PPTX: Slide text (python-pptx)
- ✅ Tables: Multi-strategy (pdfplumber)
- ❌ DOCX: Not supported
- ❌ OCR: Not supported
- ❌ HTML/Markdown: Not supported

### Enhancement Options

#### A. OCR for Scanned PDFs

**Tools:**
- **Tesseract** (OSS): Good accuracy, free, self-hosted
- **AWS Textract** (API): Excellent accuracy, $1.50/1K pages
- **Google Document AI**: Very good, $1.50/1K pages

**When Needed:**
- Scanned datasheets (older docs)
- Image-only PDFs

**Recommendation**:
✅ **Add Tesseract** - covers 90% of use cases, free
🔄 **Upgrade to Textract** if accuracy critical and budget allows

---

#### B. DOCX Extraction

**Tools:**
- python-docx (simple, structure-preserving)
- docx2txt (fast, text-only)

**Effort**: Low (1-2 hours)

**Value**:
- ✅ High demand (corporate Word docs)
- ✅ Common in enterprise environments

**Recommendation**: ✅ **High priority** - low effort, common format

---

#### C. HTML/Markdown Extraction

**Tools:**
- BeautifulSoup (HTML)
- trafilatura (web articles)
- html2text (HTML → Markdown)

**Use Case**:
- Web documentation
- README files
- Developer docs (Markdown)

**Effort**: Low-Medium (2-3 hours)

**Recommendation**: ✅ **Medium priority** - useful for web docs

---

#### D. Layout Analysis

**Tools:**
- unstructured.io (unified API)
- layout-parser (deep learning)
- PDFMiner (structure analysis)

**Use Case**:
- Complex multi-column layouts
- Preserve reading order
- Better chunk boundaries

**Effort**: Medium-High

**Recommendation**: ⏸️ **Defer** - only if current chunking inadequate

---

#### E. Image/Figure Analysis

**Tools:**
- CLIP (image embeddings)
- BLIP (image captioning)
- GPT-4V (vision model)

**Use Case**:
- Search across figures
- Answer visual questions
- Extract info from diagrams

**Effort**: High (integration + cost)

**Recommendation**: ⏸️ **Future (Phase 3+)** - high effort, niche value

---

### Document Processing Priority

1. ✅ **DOCX extraction** (High value, low effort)
2. ✅ **Tesseract OCR** (Common need, moderate effort)
3. ✅ **HTML/Markdown** (Useful for web docs)
4. ⏸️ **Layout analysis** (Only if chunking issues)
5. ⏸️ **Image analysis** (Future, Phase 3+)

---

## 6. Async Architecture Patterns

### Current State
- LLM client: httpx async + tenacity retry ✅
- SQLite: aiosqlite ✅
- Limited batch processing

### Enhancement Areas

#### A. Concurrent Document Processing

**Pattern**:
```python
import asyncio

async def process_batch(docs: List[Path]) -> List[Result]:
    tasks = [process_document(doc) for doc in docs]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

**Benefit**: 5-10x speedup on multi-doc ingestion

**Consideration**: Rate limiting, resource constraints

**Recommendation**: ✅ **Implement immediately** - low effort, high value

---

#### B. Streaming LLM Responses

**Pattern**: SSE (Server-Sent Events) for UI

**Benefit**: Better UX for long responses

**Complexity**: Error handling more difficult

**Recommendation**: ⏸️ **Optional** - UX enhancement only

---

#### C. Task Queue Integration

**Tools**:
- **Celery** (mature, Redis/RabbitMQ backend)
- **Dramatiq** (simpler, Redis/RabbitMQ)
- **Temporal** (workflow orchestration)

**When Needed**: Processing > 1000 docs/day

**Benefit**: Decoupled, scalable, fault-tolerant

**Recommendation**: ⏸️ **Defer** until scale requires it

---

#### D. Rate Limiting

**Tools**:
- aiolimiter (token bucket)
- pyrate-limiter (multiple strategies)

**Use Case**:
- API cost control
- Server protection
- Multi-tenant fairness

**Recommendation**: ✅ **Implement** if using paid APIs (Cohere, OpenAI)

---

#### E. Connection Pooling

**Current**: Single httpx client instance

**Enhancement**: Connection pool tuning

**Benefit**: Better throughput for concurrent requests

**Recommendation**: ⏸️ **Defer** - current approach adequate for < 100 req/sec

---

### Async Architecture Priority

1. ✅ **Concurrent batch processing** (Immediate value)
2. ✅ **Rate limiting** (If using paid APIs)
3. ⏸️ **Task queue** (When > 1000 docs/day)
4. ⏸️ **Streaming** (UX enhancement)
5. ⏸️ **Connection pooling** (High throughput scenarios)

---

## 7. Blind Spot Analysis

### Identified Gaps

#### 1. Observability ❌

**Missing**:
- Metrics collection (latency, success rate)
- Distributed tracing
- Structured logging (partial)

**Impact**: Medium (hard to debug production issues)

**Recommendation**:
- Add: structlog for structured logging ✅
- Add: OpenTelemetry when moving to production scale
- Add: Prometheus metrics for key operations

---

#### 2. Testing Coverage ⚠️

**Current**: Smoke tests only

**Missing**:
- Unit tests
- Integration tests
- Performance benchmarks

**Recommendation**:
- Phase 3: Add pytest suite with >80% coverage
- Add: Integration tests for storage backends
- Add: Benchmark suite for embeddings, search

---

#### 3. Error Recovery ⚠️

**Current**: Retry logic in LLM client

**Missing**:
- Circuit breaker pattern
- Graceful degradation
- Fallback strategies

**Recommendation**:
- Add: Circuit breaker (pybreaker or tenacity)
- Add: Health checks for external services
- Implement: Fallback to cache when vector store down

---

#### 4. Security ⚠️

**Current**:
- PII redaction ✅
- Path validation ✅

**Missing**:
- Prompt injection detection
- Rate limiting per user
- Input sanitization

**Recommendation**:
- Add: Prompt validation (check for injection patterns)
- Add: Per-user rate limiting (if multi-tenant)
- Review: Input sanitization for file uploads

---

#### 5. Cost Optimization ❌

**Missing**:
- Usage tracking
- Cost attribution
- Query result caching

**Recommendation**:
- Add: Usage tracking module (API calls, embeddings)
- Add: Redis cache layer for expensive operations
- Monitor: Cost per query, cost per user

---

#### 6. Scalability ⚠️

**Current**: Single-instance architecture

**Missing**:
- Horizontal scaling
- Load balancing
- Multi-region deployment

**Recommendation**:
⏸️ **Defer** until:
- > 10 concurrent users
- > 100 req/sec
- Uptime SLA requirements

---

## 8. Target Application Fit

### Application 1: mchp-MiNi-fpga-search

**Coverage**: ✅ Excellent
- Core features well-covered
- Production proven (80% P@3)

**Gaps**:
- ⚠️ Evaluation framework (in MiNi but not extracted)
- ⚠️ Web UI (domain-specific, intentionally not extracted)

**Recommendation**:
- Extract batch_eval.py → evaluation/ module (Phase 2C+)
- Keep UI in application

---

### Application 2: datasheet-review

**Coverage**: ✅ Good
- Phase 2C covers terminology + validation ✅
- Pattern library not extracted (institutional learning)
- Knowledge base not extracted (RAG enhancement)

**Gaps**:
- ⚠️ Language tools (grammar, spell check) - domain-specific
- ⚠️ Pattern library (learning from reviews)
- ⚠️ Diff/comparison tools

**Recommendation**:
- Consider pattern_library + knowledge_base (Phase 2F)
- Language tools: Keep in datasheet-review (domain-specific)

---

### Application 3: fpga_mcp

**Coverage**: ✅ Excellent
- Core features covered
- Orchestration pending (Phase 2E)

**Gaps**:
- ⚠️ Orchestration (Phase 2E will cover)

**Recommendation**:
- Phase 2E orchestration module (on roadmap)

---

### Application 4: Future MCP Servers

**Potential Gaps**:
- Authentication/authorization
- Multi-tenant isolation
- Webhook/callback patterns
- Streaming responses

**Recommendation**:
- Create MCP integration guide
- Build reference implementation

---

## 9. Decision Matrix

### Immediate Actions (Phase 2C-2E)

| Feature | Effort | Value | Priority | Status |
|---------|--------|-------|----------|--------|
| Terminology analysis | 3h | High | ✅ | **Phase 2C** |
| Completeness validation | 3h | High | ✅ | **Phase 2C** |
| Output generation | 2h | Medium | 🟡 | Phase 2D |
| Orchestration | 4h | High | 🟡 | Phase 2E |
| DOCX extraction | 2h | High | ✅ | Add to extractors |
| Abbreviation expansion | 1h | Medium | ✅ | Add to analysis |
| Concurrent processing | 2h | High | ✅ | Add to ingestion |

---

### Future Enhancements (Phase 3+)

| Feature | Effort | Value | Trigger | Phase |
|---------|--------|-------|---------|-------|
| Reranking | 4h | Medium | P@1 < 40% | 3A |
| Evaluation framework | 3h | High | Before optimization | 3A |
| Tesseract OCR | 3h | Medium | Scanned PDFs needed | 3B |
| Pattern library | 3h | Medium | Institutional learning | 3B |
| NER (custom) | 6h | Low | Have labeled data | 3C |
| Observability | 4h | High | Production deployment | 3D |
| Testing suite | 8h | High | Before v1.0 | 3D |

---

### Deferred/Not Recommended

| Feature | Reason |
|---------|--------|
| Dependency parsing | High effort, unclear ROI |
| Image analysis | Niche value, high cost |
| Fine-tuned embeddings | Need 500+ labels |
| Alternative vector stores | Qdrant proven sufficient |
| Task queues | Not needed at current scale |

---

## 10. Recommendations Summary

### Immediate (This Session)
✅ **Phase 2C**: Terminology + validation (DONE)
✅ **TECH_STACK.md**: Research doc (THIS FILE)

### Next Session (Phase 2D-2E)
🟡 **Phase 2D**: Output generation + filtering
🟡 **Phase 2E**: Orchestration module
✅ **DOCX extraction**: Add to extractors
✅ **Abbreviation expansion**: Add to analysis

### Future (Phase 3)
⏸️ **Evaluation framework**: Before optimizations
⏸️ **Reranking**: If P@1 < 40%
⏸️ **Observability**: For production deployment
⏸️ **Testing**: Unit + integration tests

### Not Recommended
❌ Fine-tuning (unless 500+ labels)
❌ Dependency parsing (unclear ROI)
❌ Image analysis (niche, expensive)
❌ Alternative vector stores (Qdrant sufficient)

---

## 11. References

- **MiNi Production Metrics**: 10K+ docs, 80% P@3, 39ms latency
- **Qdrant Docs**: https://qdrant.tech/documentation/
- **sentence-transformers**: https://www.sbert.net/
- **Reranking Research**: https://arxiv.org/abs/2104.08663

---

**Last Updated**: 2025-10-09
**Next Review**: After Phase 2E completion
