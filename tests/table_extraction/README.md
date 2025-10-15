# Table Extraction Testing Framework

This directory contains the testing framework for validating table extraction quality from PDF documents.

## Purpose

Provide quantitative metrics to assess table extraction accuracy across different tools and approaches. This framework helps:

1. **Validate consensus engine confidence scores** - Verify that high confidence correlates with high accuracy
2. **Compare extractor performance** - Identify which tools work best for different table types
3. **Detect regressions** - Ensure changes don't degrade extraction quality
4. **Build trust** - Provide evidence-based metrics for stakeholders

## Directory Structure

```
tests/table_extraction/
├── README.md                    # This file
├── ground_truth/                # Ground truth annotations
│   ├── simple_table.json        # Example: basic bordered table
│   ├── complex_table.json       # Example: merged cells, multi-level headers
│   ├── sparse_table.json        # Example: many empty cells
│   └── ...                      # Additional samples
├── sample_pdfs/                 # Test PDF files (not committed)
│   ├── simple_table.pdf
│   ├── complex_table.pdf
│   └── ...
└── test_table_accuracy.py       # Test suite
```

## Ground Truth Format

Ground truth files are JSON documents that specify the expected table structure and content for a given PDF page.

### Schema

```json
{
  "pdf_filename": "example.pdf",
  "page_num": 0,
  "description": "Human-readable description of this test case",
  "table_index": 0,
  "metadata": {
    "complexity": "simple|medium|complex|very_complex",
    "has_merged_cells": false,
    "has_multi_level_headers": false,
    "table_type": "data|specifications|timeline|other"
  },
  "expected_table": {
    "rows": 5,
    "columns": 3,
    "data": [
      ["Header 1", "Header 2", "Header 3"],
      ["Row 1, Col 1", "Row 1, Col 2", "Row 1, Col 3"],
      ["Row 2, Col 1", "Row 2, Col 2", "Row 2, Col 3"],
      ...
    ]
  },
  "validation_rules": {
    "exact_match_required": false,
    "allow_whitespace_differences": true,
    "critical_cells": [[0, 0], [1, 1]],
    "min_cell_accuracy": 0.95
  }
}
```

### Field Definitions

- **pdf_filename**: Name of the PDF file in `sample_pdfs/` directory
- **page_num**: 0-indexed page number containing the table
- **description**: Brief explanation of what this test validates
- **table_index**: 0-indexed table number on the page (if multiple tables exist)
- **metadata**: Classification of table characteristics
  - **complexity**: Enum from `TableComplexity`
  - **has_merged_cells**: Boolean indicating merged cell presence
  - **has_multi_level_headers**: Boolean for multi-row headers
  - **table_type**: Category of table content
- **expected_table**: The correct table structure
  - **rows**: Expected row count
  - **columns**: Expected column count
  - **data**: 2D array of expected cell values (strings)
- **validation_rules**: How to compare extracted vs expected
  - **exact_match_required**: If true, cell values must match exactly (case-sensitive)
  - **allow_whitespace_differences**: If true, ignore leading/trailing whitespace
  - **critical_cells**: List of [row, col] coordinates that MUST be correct
  - **min_cell_accuracy**: Minimum required cell-level accuracy (0.0-1.0)

## Creating Ground Truth

### Step 1: Extract Table Manually

1. Open the PDF in a viewer
2. Locate the table you want to test
3. Manually copy the table to a spreadsheet or text editor
4. Verify the structure (row count, column count)
5. Note any special characteristics (merged cells, complex headers, etc.)

### Step 2: Create JSON File

1. Copy `ground_truth/template.json` to a new file (e.g., `my_table.json`)
2. Fill in the metadata fields
3. Enter the expected table data as a 2D array
4. Set validation rules based on the table complexity

### Step 3: Place PDF in sample_pdfs/

1. Copy the source PDF to `sample_pdfs/`
2. Name it to match `pdf_filename` in the JSON

### Step 4: Run Tests

```bash
pytest tests/table_extraction/test_table_accuracy.py -v
```

## Metrics Computed

### Cell-Level Metrics

- **Cell Accuracy**: Percentage of cells that match exactly or fuzzy
- **Critical Cell Accuracy**: Accuracy for cells marked as critical
- **Empty Cell Precision**: How often extracted empty cells are truly empty
- **Empty Cell Recall**: How often truly empty cells are extracted as empty

### Structure Metrics

- **Row Count Accuracy**: Does extracted table have correct number of rows?
- **Column Count Accuracy**: Does extracted table have correct number of columns?
- **Structure F1**: Harmonic mean of row and column accuracy

### Overall Metrics

- **Extraction Success Rate**: Percentage of tables extracted without errors
- **Consensus Correlation**: Do high confidence scores correlate with high accuracy?

## Interpreting Results

### Accuracy Thresholds

- **≥ 0.95**: Excellent - Production ready
- **0.90-0.95**: Good - Minor issues acceptable
- **0.80-0.90**: Fair - Review recommended
- **< 0.80**: Poor - Manual verification required

### Confidence vs Accuracy

Expected correlation:
- Confidence > 0.85 → Accuracy > 0.90
- Confidence 0.70-0.85 → Accuracy 0.80-0.95
- Confidence < 0.70 → Accuracy variable (may be high or low)

## Best Practices

1. **Diverse Test Cases**: Include simple, medium, and complex tables
2. **Representative Samples**: Use real documents from your domain
3. **Edge Cases**: Test borderless tables, rotated text, merged cells
4. **Regression Testing**: Run tests after code changes
5. **Document Failures**: When tests fail, add notes about why and how to fix

## Example Usage

```python
from mchp_mcp_core.extractors import TableConsensusEngine
import json

# Load ground truth
with open("ground_truth/example.json") as f:
    gt = json.load(f)

# Extract with consensus
engine = TableConsensusEngine()
result = engine.extract_with_consensus(
    pdf_path=f"sample_pdfs/{gt['pdf_filename']}",
    page_num=gt["page_num"]
)

# Find matching table
match = result.matches[gt["table_index"]]

# Compare
accuracy = compute_cell_accuracy(
    extracted=match.best_version.data,
    expected=gt["expected_table"]["data"]
)

print(f"Confidence: {match.confidence:.2f}")
print(f"Accuracy: {accuracy:.2f}")
```

## Troubleshooting

### No tables extracted
- Check if PDF has selectable text (not scanned image)
- Try different extractors (pdfplumber, camelot, pymupdf)
- Inspect PDF structure with `pdfplumber.open(pdf).pages[0].debug_tablefinder()`

### Low accuracy on correct-looking tables
- Check whitespace normalization settings
- Review validation rules (too strict?)
- Examine cell-by-cell differences

### High confidence but low accuracy
- This indicates a bug in confidence scoring
- Review consensus matching logic
- Check if extractors are producing consistent but wrong results

## Contributing

When adding new test cases:

1. Choose representative examples from real use cases
2. Document table characteristics in metadata
3. Set reasonable validation rules
4. Add comments explaining non-obvious cases
5. Update this README if new patterns emerge
