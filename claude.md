# Claude Code Instructions for mchp-mcp-core

## Project Overview

`mchp-mcp-core` is a shared Python library providing reusable components for Microchip MCP servers and RAG applications. The library emphasizes modular design, production-ready code, and maintainability.

---

## 🔄 CRITICAL: Keep Architecture Documentation Current

**MANDATORY RULE**: Whenever you make changes to the codebase, you MUST update `docs/ARCHITECTURE.md` to reflect those changes.

### What to Update in ARCHITECTURE.md

Update immediately after:
- ✅ Adding new modules or files
- ✅ Modifying module responsibilities or APIs
- ✅ Changing data flow or integration patterns
- ✅ Refactoring package structure
- ✅ Adding/removing dependencies
- ✅ Implementing new features or capabilities

### How to Update

1. **After making code changes**, read `docs/ARCHITECTURE.md`
2. **Identify outdated sections** that no longer match the code
3. **Update the relevant sections** with current state
4. **Add new sections** if you introduced new capabilities
5. **Keep it concise** - focus on structure and data flow, not implementation details

**Example**: If you add a new extractor module, update the "Extractors Module" section with the new component and its purpose.

---

## Project Structure

```
mchp-mcp-core/
├── mchp_mcp_core/          # Core library modules
│   ├── extractors/         # Document extraction (PDF, PPTX, DOCX, tables)
│   ├── storage/            # Vector stores (Qdrant, ChromaDB), SQLite cache
│   ├── embeddings/         # sentence-transformers wrapper
│   ├── llm/                # LLM client with retry logic
│   ├── security/           # PII redaction, path validation
│   └── utils/              # Config, logging, models
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation and planning notes
├── examples/               # Example scripts and usage patterns
├── test_datasheets/        # Test PDFs for development
├── manual_review/          # Manual verification workspace
│   ├── screenshots/        # PDF screenshots of extracted tables
│   ├── extracted_tables/   # Extracted data in CSV format
│   └── ground_truth/       # Manually verified correct extractions
└── README.md               # User-facing documentation
```

---

## File Placement Rules

**ALWAYS follow these rules** - do not clutter the top-level directory:

### ✅ DO: Place files in correct locations
- **Test scripts** (`test_*.py`, `check_*.py`) → `tests/`
- **Planning docs** (session reports, phase plans) → `docs/`
- **Architecture docs** (ARCHITECTURE.md) → `docs/`
- **Example code** (usage demonstrations) → `examples/`
- **Source code** (modules, classes) → `mchp_mcp_core/`
- **Manual review materials** (screenshots, CSV exports, ground truth) → `manual_review/`

### ❌ DON'T: Create files in top-level
- Top-level should only contain: `README.md`, `pyproject.toml`, `.gitignore`, `claude.md`
- No test scripts, no planning docs, no temporary files
- Exception: `manual_review/` and `test_datasheets/` are allowed for workflow purposes

**If you catch yourself creating a file in top-level, STOP and move it to the correct directory.**

---

## Development Workflow

### 1. Planning Phase
- Use TodoWrite tool to track tasks
- For complex features, create a plan in `docs/` before coding
- Update `docs/ARCHITECTURE.md` with planned changes

### 2. Implementation Phase
- Write modular, well-documented code
- Follow existing patterns (e.g., extractors return dataclasses)
- Add type hints to all functions
- Include docstrings with examples

### 3. Testing Phase
- Create test scripts in `tests/`
- Test with real data from `test_datasheets/`
- Verify integration with existing modules

### 4. Documentation Phase ⚠️ **CRITICAL**
- Update `docs/ARCHITECTURE.md` with changes made
- Update `README.md` if user-facing features changed
- Move any planning docs to `docs/`

---

## Code Style Guidelines

### Imports
```python
# Standard library
from pathlib import Path
from typing import List, Optional

# Third-party
import fitz  # PyMuPDF

# Internal
from mchp_mcp_core.extractors import ExtractedTable
from mchp_mcp_core.utils import get_logger
```

### Error Handling
```python
try:
    result = extractor.extract(pdf_path)
except Exception as e:
    logger.error(f"Extraction failed: {e}")
    return default_value
```

### Logging
```python
from mchp_mcp_core.utils import get_logger

logger = get_logger(__name__)

logger.debug("Detailed diagnostic info")
logger.info("High-level progress updates")
logger.warning("Recoverable issues")
logger.error("Failures requiring attention")
```

---

## Module Responsibilities

See `docs/ARCHITECTURE.md` for detailed module breakdown and data flow.

**Quick Reference**:
- `extractors/` - Extract structured data from documents
- `storage/` - Persist and retrieve data (vector + cache)
- `embeddings/` - Generate vector embeddings
- `llm/` - LLM API client with retry/rate limiting
- `security/` - PII redaction, path validation
- `utils/` - Shared utilities (config, logging, models)

---

## Testing Strategy

### Test File Organization
```
tests/
├── smoke_test.py              # Quick sanity check for all modules
├── test_*.py                  # Specific feature tests
├── check_*.py                 # Diagnostic/inspection scripts
└── table_extraction/          # Domain-specific test suites
```

### Running Tests
```bash
# From project root
export PYTHONPATH=/home/jorgill/mchp-mcp-core
python tests/smoke_test.py
python tests/test_multipage_detection.py
```

---

## Common Patterns

### Extractor Pattern
All extractors return typed results:
```python
@dataclass
class ExtractionResult:
    success: bool
    tables: List[ExtractedTable]
    error: Optional[str] = None
```

### Configuration Pattern
Use Pydantic settings for config:
```python
from mchp_mcp_core.utils import BaseConfig

class MyConfig(BaseConfig):
    api_key: str
    timeout: int = 30
```

### Logging Pattern
Always use module-level logger:
```python
logger = get_logger(__name__)
logger.info(f"Processing {item_count} items")
```

---

## Dependencies

- **PyMuPDF (fitz)**: Fast PDF parsing and rendering
- **pdfplumber**: Text-based table extraction
- **camelot-py**: Line-based table extraction
- **sentence-transformers**: Embedding models
- **qdrant-client**: Vector database
- **Pydantic**: Configuration and validation
- **tenacity**: Retry logic for LLM calls

---

## Remember

1. ✅ **Keep docs/ARCHITECTURE.md current** after every code change
2. ✅ **Place files in correct directories** (tests/, docs/, examples/)
3. ✅ **Follow existing patterns** - check similar code before implementing
4. ✅ **Use TodoWrite** to track progress on complex tasks
5. ✅ **Test with real data** from test_datasheets/

**When in doubt, check `docs/ARCHITECTURE.md` for current state and patterns.**
