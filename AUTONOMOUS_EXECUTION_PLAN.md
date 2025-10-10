# Autonomous Execution Plan: Phases C → B → A → D

**Start Time**: 2025-10-09 Evening
**Completion Target**: Morning report
**Status**: IN PROGRESS

---

## Execution Order

**C** → Quick Wins (DOCX, abbreviations, async)
**B** → Orchestration
**A** → Output generation
**D** → Evaluation framework

---

## Phase C: Quick Wins (~1.5h)

### C1: DOCX Extraction
- **File**: `extractors/docx.py` (150 LOC)
- **Lib**: python-docx
- **Features**: Paragraphs, tables, styles → ExtractedChunk

### C2: Abbreviation Expansion
- **File**: `analysis/abbreviations.py` (100 LOC)
- **Dict**: SPI → Serial Peripheral Interface (100+ entries)
- **Use**: Improve search recall

### C3: Async Batch Processing
- **File**: `utils/async_batch.py` (80 LOC)
- **Pattern**: asyncio.gather with concurrency limit
- **Benefit**: 5-10x speedup

**Commit**: v0.1.4

---

## Phase B: Orchestration (~2h)

- **File**: `ingestion/orchestrator.py` (400 LOC)
- **Source**: MiNi + fpga_mcp patterns
- **Features**:
  - Directory scanning + manifest deduplication
  - Job queue with priorities
  - Rich progress tracking
  - Report generation (HTML + JSON)

**Commit**: v0.1.5

---

## Phase A: Output Generation (~1.5h)

- **Files**:
  - `output/markdown.py` (350 LOC)
  - `output/filter.py` (100 LOC)
- **Source**: datasheet-review output modules
- **Features**:
  - Document reassembly
  - TOC generation
  - Change tracking
  - Content filtering

---

## Phase D: Evaluation (~1h)

- **File**: `evaluation/metrics.py` (300 LOC)
- **Source**: MiNi batch_eval.py
- **Metrics**: P@1, P@3, P@5, MRR
- **Use**: Benchmark before optimizations

**Commit**: v0.1.6 (A+D together)

---

## Final Deliverables

### New Modules: 7 files, ~1,480 LOC
### New Categories: ingestion/, output/, evaluation/
### Version: 0.1.3 → 0.1.6
### Dependencies: +python-docx, +rich

---

## Morning Report Location

`SESSION_REPORT_PHASE_2C_TO_2D.md` will contain:
- ✅ Completion status per phase
- 📊 Statistics (LOC, modules, commits)
- ⚠️ Any errors encountered
- 🎯 Next steps
- 📝 Usage examples

---

**Status**: Executing...
