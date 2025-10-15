# Sample PDFs for Table Extraction Testing

This directory should contain sample PDF files for table extraction testing.

## Setup

1. Add PDF files that correspond to ground truth annotations in `../ground_truth/`
2. Name PDFs to match the `pdf_filename` field in ground truth JSON files

## Example

If you have `ground_truth/simple_specifications.json` with:
```json
{
  "pdf_filename": "simple_specifications.pdf",
  ...
}
```

Then place the PDF at: `sample_pdfs/simple_specifications.pdf`

## Creating Test PDFs

You can create simple test PDFs using:

1. **LaTeX**: Create tables and compile to PDF
2. **Microsoft Word/LibreOffice**: Create tables and export to PDF
3. **Python reportlab**: Generate PDFs programmatically
4. **Real documents**: Extract pages from real datasheets (ensure you have rights to use them)

## Best Practices

- Use realistic table layouts from your domain
- Include variety: simple bordered tables, complex layouts, sparse tables
- Keep file sizes small (<1MB preferred)
- Document the source and characteristics in the ground truth JSON

## Note

PDF files in this directory are not committed to git (see .gitignore).
Store important test files in a shared location if needed by the team.
