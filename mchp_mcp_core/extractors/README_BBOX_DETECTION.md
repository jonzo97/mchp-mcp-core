# Table Bounding Box Detection Strategies

## Problem Statement

**Issue Discovered**: 2025-10-20

When extracting tables from PDFs, different extractors report different bounding boxes (bboxes) for the same table, leading to:
- Misaligned row comparisons in consensus metrics
- Cropped/incomplete table screenshots
- Inaccurate data row accuracy metrics (0.48 instead of ~1.00)

**Specific Example (Page 55)**:
```
pdfplumber: No bbox (uses full page)
pymupdf:    BBox: (51.2, 69.5, 560.9, 555.3)   ✓ CORRECT
camelot:    BBox: (44.3, 228.4, 569.7, 746.5)  ✗ WRONG (159 points too low!)
```

Camelot's bbox starts at Y=228 but its CSV contains PA0 which actually appears at ~Y=69, proving the bbox is inaccurate.

---

## Research Findings

### Community Best Practices

**Two-Stage Detection** is the standard approach:
1. Use one tool to detect table regions (bboxes)
2. Pass those regions to extractors for content extraction

**Visual Debugging** is critical:
```python
camelot.plot(tables[0], kind='contour')  # Visualize detected region
```

### Known Issues

- **Camelot GitHub Issue #486**: "The extracted table box coordinates do not correspond to the images converted from the PDF"
  - This is exactly our problem!
  - Community solution: Use visual debugging + two-stage workflow

- **Camelot Stream Mode**: Bbox detection can be unreliable for borderless tables
  - `edge_tol` parameter affects detection (default: 50)
  - Larger values improve detection for text placed far apart

### Available Tools

| Tool | Approach | Pros | Cons |
|------|----------|------|------|
| **PyMuPDF** | Rule-based (lines/rectangles) | Fast, accurate for structured tables | May miss borderless tables |
| **Camelot Lattice** | Rule-based (line detection) | Good for gridded tables | Requires visible borders |
| **Camelot Stream** | Rule-based (text spacing) | Works without borders | Bbox can be inaccurate |
| **pdfplumber** | Rule-based (text/lines) | Great for complex tables | No auto bbox (uses full page) |
| **img2table** | OpenCV-based | Specialized for borderless tables | New dependency |
| **Table Transformer** | Deep learning (DETR) | Most robust, handles anything | Heavy (PyTorch), slow, overkill |

---

## Recommended Approach: **Consensus Bbox (Union Strategy)**

### Most Robust Solution

**Run all 3 extractors, take the UNION of their bboxes:**

```python
def detect_table_region_consensus(pdf_path: str, page_num: int) -> Tuple[float, float, float, float]:
    """
    Detect table region using consensus of all extractors.

    Strategy:
    1. Run all 3 extractors to detect tables
    2. For each table, compute union bbox (min x0, min y0, max x1, max y1)
    3. Add padding for safety
    4. Use unified region for all extractors

    Why union?
    - If one extractor's bbox is too small (like Camelot), others expand it
    - If one extractor misses part of table, others catch it
    - Not dependent on any single extractor being perfect
    """
    bboxes = []

    # Get bboxes from each extractor
    pymupdf_tables = pymupdf_extractor.extract_tables(pdf_path, page_num)
    for table in pymupdf_tables.tables:
        if table.bbox:
            bboxes.append(table.bbox)

    camelot_tables = camelot_extractor.extract_tables(pdf_path, page_num)
    for table in camelot_tables.tables:
        if table.bbox:
            bboxes.append(table.bbox)

    # pdfplumber doesn't provide bbox by default, but we can use page.find_tables()
    # and get bbox from the detected table objects

    if not bboxes:
        return None  # Fallback to full page

    # Compute union: (min_x0, min_y0, max_x1, max_y1)
    x0 = min(bbox[0] for bbox in bboxes)
    y0 = min(bbox[1] for bbox in bboxes)
    x1 = max(bbox[2] for bbox in bboxes)
    y1 = max(bbox[3] for bbox in bboxes)

    # Add 10-point padding for safety
    PADDING = 10
    return (x0 - PADDING, y0 - PADDING, x1 + PADDING, y1 + PADDING)
```

### Implementation Steps

1. **Modify ConsensusEngine.extract_with_consensus()**:
   ```python
   # Step 1: Run all extractors to detect bboxes
   union_bbox = detect_table_region_consensus(pdf_path, page_num)

   # Step 2: Pass unified bbox to all extractors
   results = []
   for extractor in extractors:
       result = extractor.extract_tables(pdf_path, page_num, region=union_bbox)
       results.append(result)

   # Step 3: Run consensus on aligned tables
   matches = match_tables_by_position(results, union_bbox)
   ```

2. **Add `region` parameter to all extractors**:
   ```python
   @abstractmethod
   def extract_tables(
       self,
       pdf_path: str | Path,
       page_num: int,
       region: Optional[Tuple[float, float, float, float]] = None,
       **kwargs
   ) -> ExtractionResult:
       """
       Args:
           region: Optional bbox (x0, y0, x1, y1) to restrict search area
       """
   ```

3. **Add visual debugging**:
   ```python
   # After detecting union bbox, visualize it
   camelot.plot(tables[0], kind='contour')  # Show detected region
   ```

### Why Union Over Alternatives?

| Strategy | Pros | Cons |
|----------|------|------|
| **Union (RECOMMENDED)** | Captures full table, handles outliers gracefully | Might include extra whitespace |
| Single extractor (PyMuPDF) | Simple, fast | Single point of failure |
| Intersection | Only areas all agree on | Misses edges if extractors disagree |
| Voting/median | Outliers don't dominate | Complex, still might miss edges |

---

## Alternative Approaches

### Option A: PyMuPDF Only (Simplest)
- **Use case**: PyMuPDF bbox is accurate for your PDFs
- **Implementation**: 30 minutes
- **Risk**: Single point of failure

### Option B: Camelot Two-Stage (Camelot-specific)
- **Use case**: Only using Camelot
- **Pattern**: Use lattice mode to get bbox, pass to stream mode
- **Implementation**: 20 minutes
- **Risk**: Still dependent on Camelot

### Option C: img2table (New Tool)
- **Use case**: Many borderless tables
- **Implementation**: 2-3 hours (new dependency integration)
- **Pros**: Specialized for borderless tables
- **Cons**: Another dependency to maintain

### Option D: Table Transformer (Future-Proof)
- **Use case**: Extremely complex/scanned PDFs
- **Implementation**: 4-6 hours + GPU setup
- **Pros**: State-of-the-art deep learning
- **Cons**: Heavy, slow, overkill for most cases

---

## Testing Strategy

**After implementing consensus bbox:**

1. **Test on page 55** (the problem case):
   ```python
   # Should show union bbox includes PA0 through PA31
   union_bbox = detect_table_region_consensus(pdf_path, page_num=55)
   print(f"Union bbox: {union_bbox}")

   # Camelot's screenshot should no longer be cropped
   generate_screenshot_with_bbox(pdf_path, 55, union_bbox, "camelot")
   ```

2. **Verify data row accuracy improves**:
   ```bash
   python tests/check_page55_harmony.py
   # Expected: Data row accuracy ~0.95+ (was 0.48)
   ```

3. **Test on pages with multiple tables**:
   - Verify union strategy doesn't merge separate tables
   - Ensure each table gets its own bbox

4. **Edge cases to test**:
   - Page with no tables (should fallback gracefully)
   - Page with 2+ tables (should detect multiple bboxes)
   - Fully borderless tables (verify union captures them)
   - Tables with merged cells (verify no cropping)

---

## Future Enhancements

1. **Add visual debugging dashboard**:
   - Show bboxes from each extractor overlaid on PDF
   - Highlight union bbox in different color
   - Display confidence scores

2. **Add bbox validation**:
   - Flag if bboxes differ by >100 points (likely error)
   - Warn if union bbox is >2x size of any individual bbox
   - Detect if bboxes don't overlap at all (separate tables)

3. **Add Table Transformer fallback**:
   - If rule-based extractors disagree significantly (>50pt diff)
   - Run TATR as tiebreaker
   - Cache TATR results to avoid re-running

4. **Optimize performance**:
   - Currently runs all extractors twice (once for bbox, once for content)
   - Could cache first extraction and reuse results
   - Estimated 2x speedup

---

## References

- **Camelot Issue #486**: [bbox coordinates don't match actual extraction](https://github.com/atlanhq/camelot/issues/486)
- **PyMuPDF Table Detection**: [blog.artifex.com](https://artifex.com/blog/table-recognition-extraction-from-pdfs-pymupdf-python)
- **Table Transformer (TATR)**: [github.com/microsoft/table-transformer](https://github.com/microsoft/table-transformer)
- **img2table**: [github.com/xavctn/img2table](https://github.com/xavctn/img2table)
- **Stack Overflow: Camelot table_areas**: [How to find table region for camelot](https://stackoverflow.com/questions/58025146/how-to-find-table-region-for-camelot)

---

**Last Updated**: 2025-10-20
**Status**: Ready for implementation
**Estimated Time**: 3-4 hours for consensus bbox approach
