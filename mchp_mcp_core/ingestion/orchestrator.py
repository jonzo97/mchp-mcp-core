"""
Ingestion orchestration module.

Coordinates document ingestion with multi-format support, progress tracking,
error handling, and report generation.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table

from mchp_mcp_core.extractors import (
    PDFExtractor,
    PPTXExtractor,
    DOCXExtractor,
    extract_metadata
)
from mchp_mcp_core.embeddings import EmbeddingModel
from mchp_mcp_core.models import ExtractedChunk
from mchp_mcp_core.storage import (
    DocumentChunk,
    IngestionReport,
    ManifestRepository,
    ManifestStatus,
    DocumentManifest,
    QdrantVectorStore,
    ChromaDBVectorStore
)
from mchp_mcp_core.utils import (
    get_logger,
    compute_checksum,
    process_batch_concurrent
)

logger = get_logger(__name__)
console = Console()


def sha256(s: str) -> str:
    """Generate SHA-256 hash of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class IngestionJob:
    """Represents a single document ingestion task."""

    path: Path
    doc_id: str
    title: str
    checksum: str
    file_type: str
    size_bytes: int
    manifest: Optional[DocumentManifest] = None


@dataclass
class IngestionResult:
    """Result of processing a single document."""

    job: IngestionJob
    chunks: List[DocumentChunk]
    success: bool
    error: Optional[str] = None
    duration_seconds: float = 0.0


class IngestionOrchestrator:
    """
    Coordinates multi-format document ingestion with parallel processing.

    Features:
    - Multi-format support (PDF, PPTX, DOCX)
    - Parallel processing with configurable concurrency
    - Manifest tracking (optional)
    - Progress display with Rich
    - Error handling and reporting
    - JSONL export and HTML report generation

    Example:
        >>> # Basic usage
        >>> orchestrator = IngestionOrchestrator(
        ...     vector_store=QdrantVectorStore(),
        ...     embedding_model=EmbeddingModel(),
        ...     max_concurrent=10
        ... )
        >>> report = await orchestrator.run(
        ...     directory="./docs",
        ...     output_jsonl="./data/corpus.jsonl",
        ...     report_html="./data/report.html"
        ... )
        >>> print(f"Processed {report.total_files} files, {report.total_chunks} chunks")

        >>> # With manifest tracking
        >>> from mchp_mcp_core.storage import ManifestRepository
        >>> repo = ManifestRepository(Path("./data/manifest.db"))
        >>> orchestrator = IngestionOrchestrator(
        ...     manifest_repo=repo,
        ...     max_concurrent=5
        ... )
        >>> jobs = orchestrator.scan_directory("./docs")
        >>> results = await orchestrator.process_jobs(jobs)
    """

    # Extractor registry mapping file extensions to extractor classes
    EXTRACTORS = {
        '.pdf': PDFExtractor,
        '.pptx': PPTXExtractor,
        '.ppt': PPTXExtractor,
        '.docx': DOCXExtractor,
        '.doc': DOCXExtractor,
    }

    DEFAULT_ALLOWED_EXTENSIONS = ['.pdf', '.pptx', '.ppt', '.docx', '.doc']

    def __init__(
        self,
        vector_store: Optional[Union[QdrantVectorStore, ChromaDBVectorStore]] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        manifest_repo: Optional[ManifestRepository] = None,
        max_concurrent: int = 5,
        max_file_size_mb: float = 50.0,
        allowed_extensions: Optional[List[str]] = None
    ):
        """
        Initialize orchestrator.

        Args:
            vector_store: Vector database for storing embeddings (optional)
            embedding_model: Model for generating embeddings (optional)
            manifest_repo: Repository for tracking document versions (optional)
            max_concurrent: Maximum concurrent processing tasks
            max_file_size_mb: Maximum file size in MB
            allowed_extensions: List of allowed file extensions (default: all supported)
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.manifest_repo = manifest_repo
        self.max_concurrent = max_concurrent
        self.max_file_size_mb = max_file_size_mb
        self.allowed_extensions = allowed_extensions or self.DEFAULT_ALLOWED_EXTENSIONS

    def scan_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True
    ) -> List[IngestionJob]:
        """
        Scan directory for documents and create ingestion jobs.

        Args:
            directory: Directory to scan
            recursive: Whether to scan recursively (default: True)

        Returns:
            List of ingestion jobs

        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        jobs = []
        pattern = "**/*" if recursive else "*"

        logger.info(f"Scanning directory: {directory}")
        logger.info(f"Allowed extensions: {', '.join(self.allowed_extensions)}")

        for file_path in directory.glob(pattern):
            # Skip directories
            if file_path.is_dir():
                continue

            # Check extension
            if file_path.suffix.lower() not in self.allowed_extensions:
                continue

            # Check file size
            size_bytes = file_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            if size_mb > self.max_file_size_mb:
                logger.warning(
                    f"Skipping {file_path.name}: "
                    f"size {size_mb:.1f}MB exceeds {self.max_file_size_mb}MB limit"
                )
                continue

            # Compute checksum
            checksum = compute_checksum(file_path)

            # Create job
            job = IngestionJob(
                path=file_path,
                doc_id=str(file_path.resolve()),
                title=file_path.stem,
                checksum=checksum,
                file_type=file_path.suffix.lower(),
                size_bytes=size_bytes
            )

            # Register with manifest if available
            if self.manifest_repo:
                manifest = DocumentManifest(
                    doc_id=job.doc_id,
                    version="1.0",  # TODO: Extract version from metadata
                    checksum=checksum,
                    size_bytes=size_bytes,
                    status=ManifestStatus.STAGED,
                    source_url=str(file_path),
                    page_count=None,
                    notes=None
                )
                self.manifest_repo.upsert(manifest)
                job.manifest = manifest

            jobs.append(job)

        logger.info(f"Found {len(jobs)} documents to process")
        return jobs

    async def process_job(self, job: IngestionJob) -> IngestionResult:
        """
        Process a single ingestion job.

        Args:
            job: Ingestion job to process

        Returns:
            Ingestion result with chunks or error
        """
        start_time = time.time()

        try:
            # Update manifest status
            if self.manifest_repo and job.manifest:
                self.manifest_repo.update_status(
                    job.checksum,
                    ManifestStatus.EXTRACTING
                )

            # Get extractor class
            extractor_class = self.EXTRACTORS.get(job.file_type)
            if not extractor_class:
                raise ValueError(f"Unsupported file type: {job.file_type}")

            # Instantiate extractor
            extractor = extractor_class()

            # Extract chunks (run in thread to avoid blocking)
            extracted_chunks: List[ExtractedChunk] = await asyncio.to_thread(
                extractor.extract_document,
                file_path=job.path,
                doc_id=job.doc_id,
                title=job.title
            )

            # Get metadata from first page/slide
            first_page_text = extracted_chunks[0].content if extracted_chunks else None
            metadata = extract_metadata(
                path=job.path,
                first_page_text=first_page_text,
                docs_root=job.path.parent
            )

            # Convert ExtractedChunk to DocumentChunk with metadata
            chunks = []
            updated_at = datetime.fromtimestamp(job.path.stat().st_mtime).isoformat()

            for chunk in extracted_chunks:
                doc_chunk = DocumentChunk(
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    source_path=str(job.path),
                    updated_at=updated_at,
                    slide_or_page=chunk.page_number,
                    chunk_id=chunk.chunk_index,
                    text=chunk.content,
                    sha256=sha256(f"{chunk.doc_id}_{chunk.page_number}_{chunk.content[:200]}"),
                    document_type=metadata.get("document_type"),
                    category_tags=metadata.get("category_tags", []),
                    product_family=metadata.get("product_family"),
                    document_date=metadata.get("document_date"),
                    version=metadata.get("version"),
                    subfolder_path=metadata.get("subfolder_path", "")
                )
                chunks.append(doc_chunk)

            # Generate embeddings and store if models available
            if self.embedding_model and self.vector_store and chunks:
                texts = [c.text for c in chunks]
                embeddings = await asyncio.to_thread(
                    self.embedding_model.encode,
                    texts
                )
                # Store in vector database
                await asyncio.to_thread(
                    self.vector_store.add_documents,
                    chunks,
                    skip_duplicates=True,
                    show_progress=False
                )

            # Update manifest status
            if self.manifest_repo and job.manifest:
                self.manifest_repo.update_status(
                    job.checksum,
                    ManifestStatus.READY,
                    notes=f"Extracted {len(chunks)} chunks"
                )

            duration = time.time() - start_time
            logger.info(
                f"✓ {job.path.name}: {len(chunks)} chunks "
                f"({duration:.2f}s)"
            )

            return IngestionResult(
                job=job,
                chunks=chunks,
                success=True,
                duration_seconds=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"✗ {job.path.name}: {e}")

            # Update manifest status
            if self.manifest_repo and job.manifest:
                self.manifest_repo.update_status(
                    job.checksum,
                    ManifestStatus.FAILED,
                    notes=str(e)
                )

            return IngestionResult(
                job=job,
                chunks=[],
                success=False,
                error=str(e),
                duration_seconds=duration
            )

    async def process_jobs(
        self,
        jobs: List[IngestionJob],
        show_progress: bool = True
    ) -> List[IngestionResult]:
        """
        Process multiple jobs concurrently with progress tracking.

        Args:
            jobs: List of ingestion jobs
            show_progress: Whether to show progress bar (default: True)

        Returns:
            List of ingestion results
        """
        if not jobs:
            logger.warning("No jobs to process")
            return []

        logger.info(f"Processing {len(jobs)} jobs with max_concurrent={self.max_concurrent}")

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Processing documents...", total=len(jobs))

                results = []
                # Process in batches to respect max_concurrent
                for i in range(0, len(jobs), self.max_concurrent):
                    batch = jobs[i:i + self.max_concurrent]
                    batch_results = await asyncio.gather(
                        *[self.process_job(job) for job in batch]
                    )
                    results.extend(batch_results)
                    progress.update(task, advance=len(batch))

                return results
        else:
            # Use utility function for concurrent processing without progress bar
            results = await process_batch_concurrent(
                jobs,
                self.process_job,
                max_concurrent=self.max_concurrent,
                show_progress=False
            )
            return results

    def generate_report(
        self,
        results: List[IngestionResult],
        start_time: float
    ) -> IngestionReport:
        """
        Generate ingestion report from results.

        Args:
            results: List of ingestion results
            start_time: Start time of ingestion (time.time())

        Returns:
            Ingestion report with statistics
        """
        duration = time.time() - start_time

        all_chunks = []
        processed_files = []
        error_files = []

        for result in results:
            if result.success:
                all_chunks.extend(result.chunks)
                processed_files.append(str(result.job.path))
            else:
                error_files.append({
                    "file": str(result.job.path),
                    "error": result.error or "Unknown error"
                })

        report = IngestionReport(
            total_files=len(results),
            total_chunks=len(all_chunks),
            duplicates_skipped=0,  # TODO: Track duplicates in vector store
            errors=len(error_files),
            duration_seconds=duration,
            files_processed=processed_files,
            error_files=error_files
        )

        return report

    def save_to_jsonl(
        self,
        results: List[IngestionResult],
        output_path: Union[str, Path]
    ):
        """
        Save chunks to JSONL file.

        Args:
            results: List of ingestion results
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_chunks = []
        for result in results:
            if result.success:
                all_chunks.extend(result.chunks)

        logger.info(f"Saving {len(all_chunks)} chunks to {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')

        logger.info(f"Saved to {output_path}")

    def generate_html_report(
        self,
        report: IngestionReport,
        output_path: Union[str, Path]
    ):
        """
        Generate HTML report with statistics and error details.

        Args:
            report: Ingestion report
            output_path: Output HTML file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ingestion Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .metric strong {{ color: #0066cc; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #0066cc; color: white; }}
        .error {{ color: #cc0000; }}
        .success {{ color: #00cc00; }}
    </style>
</head>
<body>
    <h1>Document Ingestion Report</h1>
    <p><strong>Generated:</strong> {datetime.now().isoformat()}</p>

    <div class="summary">
        <h2>Summary</h2>
        <div class="metric"><strong>Total Files Processed:</strong> {report.total_files}</div>
        <div class="metric"><strong>Total Chunks Created:</strong> {report.total_chunks}</div>
        <div class="metric"><strong>Duplicates Skipped:</strong> {report.duplicates_skipped}</div>
        <div class="metric"><strong>Errors:</strong> <span class="{'error' if report.errors > 0 else 'success'}">{report.errors}</span></div>
        <div class="metric"><strong>Duration:</strong> {report.duration_seconds:.2f} seconds</div>
        <div class="metric"><strong>Success Rate:</strong> {report.success_rate:.1f}%</div>
        <div class="metric"><strong>Avg Chunks per File:</strong> {report.chunks_per_file:.1f}</div>
    </div>

    <h2>Processed Files</h2>
    <table>
        <tr>
            <th>#</th>
            <th>File Path</th>
        </tr>
        {''.join(f'<tr><td>{i+1}</td><td>{f}</td></tr>' for i, f in enumerate(report.files_processed))}
    </table>

    {f'''<h2 class="error">Errors</h2>
    <table>
        <tr>
            <th>File</th>
            <th>Error</th>
        </tr>
        {''.join(f'<tr><td>{e["file"]}</td><td>{e["error"]}</td></tr>' for e in report.error_files)}
    </table>''' if report.error_files else ''}
</body>
</html>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Generated HTML report: {output_path}")

    def display_summary(self, report: IngestionReport):
        """
        Display summary table with Rich formatting.

        Args:
            report: Ingestion report
        """
        table = Table(title="Ingestion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Files", str(report.total_files))
        table.add_row("Total Chunks", str(report.total_chunks))
        table.add_row("Duplicates Skipped", str(report.duplicates_skipped))
        table.add_row(
            "Errors",
            str(report.errors),
            style="red" if report.errors > 0 else "green"
        )
        table.add_row("Duration", f"{report.duration_seconds:.2f}s")
        table.add_row("Success Rate", f"{report.success_rate:.1f}%")
        table.add_row("Avg Chunks/File", f"{report.chunks_per_file:.1f}")

        console.print(table)

    async def run(
        self,
        directory: Union[str, Path],
        output_jsonl: Optional[Union[str, Path]] = None,
        report_html: Optional[Union[str, Path]] = None,
        recursive: bool = True,
        show_progress: bool = True
    ) -> IngestionReport:
        """
        Run complete ingestion pipeline end-to-end.

        Steps:
        1. Scan directory for documents
        2. Process documents concurrently
        3. Generate embeddings and store (if configured)
        4. Export to JSONL (if requested)
        5. Generate HTML report (if requested)
        6. Display summary

        Args:
            directory: Directory to scan
            output_jsonl: Optional JSONL output path
            report_html: Optional HTML report path
            recursive: Whether to scan recursively (default: True)
            show_progress: Whether to show progress bar (default: True)

        Returns:
            Ingestion report with statistics
        """
        start_time = time.time()

        logger.info("=" * 50)
        logger.info("Starting document ingestion pipeline")
        logger.info("=" * 50)

        # Step 1: Scan directory
        jobs = self.scan_directory(directory, recursive=recursive)

        if not jobs:
            logger.warning("No documents found")
            return IngestionReport(
                total_files=0,
                total_chunks=0,
                duplicates_skipped=0,
                errors=0,
                duration_seconds=time.time() - start_time,
                files_processed=[],
                error_files=[]
            )

        # Step 2: Process jobs
        results = await self.process_jobs(jobs, show_progress=show_progress)

        # Step 3: Generate report
        report = self.generate_report(results, start_time)

        # Step 4: Save JSONL if requested
        if output_jsonl:
            self.save_to_jsonl(results, output_jsonl)

        # Step 5: Generate HTML report if requested
        if report_html:
            self.generate_html_report(report, report_html)

        # Step 6: Display summary
        self.display_summary(report)

        logger.info("=" * 50)
        logger.info("Ingestion complete!")
        logger.info(f"Total time: {report.duration_seconds:.2f} seconds")
        if output_jsonl:
            logger.info(f"JSONL: {output_jsonl}")
        if report_html:
            logger.info(f"Report: {report_html}")
        logger.info("=" * 50)

        return report
