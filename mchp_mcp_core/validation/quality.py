"""
Semantic Completeness Validator.

Uses LLM to extract claims from Features/Introduction sections and validates
that supporting evidence exists in the document (specs, tables, detailed sections).

This catches critical documentation gaps like:
- "USB 2.0 high-speed" claimed but no timing specs
- "AES-256 encryption" mentioned but no crypto section
- Features listed but no corresponding functional descriptions

The validator can optionally leverage a knowledge base (RAG) to enhance
suggestions with examples from approved documentation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from mchp_mcp_core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Claim:
    """
    Represents a technical claim that needs evidence.

    Attributes:
        claim_id: Unique identifier for this claim
        claim_text: The actual claim text
        claim_type: Type of claim ('feature', 'specification', 'capability')
        section: Section where claim was made
        page_number: Page number where claim appears
        requires_evidence: List of evidence types needed
        confidence: LLM confidence in extraction (0.0 to 1.0)
    """
    claim_id: str
    claim_text: str
    claim_type: str
    section: str
    page_number: int
    requires_evidence: list[str]
    confidence: float = 0.0


@dataclass
class Evidence:
    """
    Represents supporting evidence for a claim.

    Attributes:
        evidence_id: Unique identifier
        evidence_type: Type of evidence ('section', 'table', 'figure', 'specification')
        content: Evidence content (truncated for reports)
        location: Section number or page where found
        relevance_score: How well it supports the claim (0.0 to 1.0)
    """
    evidence_id: str
    evidence_type: str
    content: str
    location: str
    relevance_score: float = 0.0


@dataclass
class ValidationResult:
    """
    Result of validating a claim against evidence.

    Attributes:
        claim: The claim being validated
        evidence_found: List of evidence that supports the claim
        is_supported: Whether claim has adequate support
        confidence: Confidence in validation (0.0 to 1.0)
        severity: Issue severity ('critical', 'high', 'medium', 'low')
        suggestion: Actionable suggestion for improvement
        missing_evidence: List of missing evidence types
    """
    claim: Claim
    evidence_found: list[Evidence]
    is_supported: bool
    confidence: float
    severity: str
    suggestion: str
    missing_evidence: list[str]


class CompletenessValidator:
    """
    Validates semantic completeness of technical documentation.

    Uses an LLM to extract claims from high-level sections (features, intro)
    and searches the document for supporting evidence (specs, tables, descriptions).

    Example:
        >>> from mchp_mcp_core.llm import LLMClient
        >>>
        >>> llm = LLMClient(config)
        >>> validator = CompletenessValidator(llm_client=llm)
        >>> report = await validator.validate_completeness(chunks, structure)
        >>>
        >>> print(f"Claims: {report['total_claims']}")
        >>> print(f"Supported: {report['support_rate']}%")
        >>> for issue in report['critical_issues']:
        ...     print(f"Missing: {issue['claim']}")
    """

    # Sections that typically make claims
    DEFAULT_CLAIM_SECTIONS = [
        'introduction', 'features', 'overview', 'highlights',
        'key features', 'product overview', 'summary'
    ]

    # Evidence types to search for
    DEFAULT_EVIDENCE_TYPES = {
        'timing': ['timing diagram', 'timing specification', 'timing characteristics'],
        'electrical': ['electrical characteristics', 'dc characteristics', 'ac characteristics'],
        'pinout': ['pinout', 'pin description', 'pin configuration'],
        'functional': ['functional description', 'operation', 'module description'],
        'register': ['register', 'control register', 'status register'],
        'memory': ['memory map', 'memory organization'],
    }

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        knowledge_base: Optional[Any] = None,
        claim_sections: Optional[list[str]] = None,
        evidence_types: Optional[dict[str, list[str]]] = None,
        enabled: bool = True
    ):
        """
        Initialize completeness validator.

        Args:
            llm_client: LLM client for claim extraction (mchp_mcp_core.llm.LLMClient)
            knowledge_base: Optional knowledge base for RAG enhancement
            claim_sections: Section names where claims are typically made.
                           If None, uses DEFAULT_CLAIM_SECTIONS.
            evidence_types: Dict mapping evidence category to section keywords.
                           If None, uses DEFAULT_EVIDENCE_TYPES.
            enabled: Whether validation is enabled

        Example:
            >>> from mchp_mcp_core.llm import LLMClient
            >>> llm = LLMClient(config)
            >>> validator = CompletenessValidator(
            ...     llm_client=llm,
            ...     claim_sections=['features', 'capabilities']
            ... )
        """
        self.llm_client = llm_client
        self.knowledge_base = knowledge_base
        self.enabled = enabled and llm_client is not None

        self.claim_sections = claim_sections or self.DEFAULT_CLAIM_SECTIONS
        self.evidence_types = evidence_types or self.DEFAULT_EVIDENCE_TYPES

    async def validate_completeness(
        self,
        chunks: list[Any],
        document_structure: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Main validation method.

        Args:
            chunks: Document chunks with 'content', 'section_hierarchy',
                   'page_start', 'chunk_id', 'chunk_type' attributes
            document_structure: Optional document metadata and structure

        Returns:
            Completeness validation report dict with keys:
                - enabled: Whether validation was run
                - total_claims: Number of claims extracted
                - claims_supported: Number with adequate evidence
                - claims_unsupported: Number lacking evidence
                - support_rate: Percentage of supported claims
                - critical_issues: List of critical unsupported claims
                - high_priority_issues: List of high priority issues
                - summary: Count dict by severity level

        Example:
            >>> report = await validator.validate_completeness(chunks)
            >>> if report['support_rate'] < 80:
            ...     print("Documentation may be incomplete")
            >>> for issue in report['critical_issues']:
            ...     print(f"Add: {', '.join(issue['missing_evidence'])}")
        """
        if not self.enabled or not self.llm_client:
            logger.info("Completeness validation disabled (no LLM client)")
            return {
                'enabled': False,
                'claims_extracted': 0,
                'claims_supported': 0,
                'critical_issues': []
            }

        logger.info("Starting semantic completeness validation...")

        # Step 1: Extract claims from intro/features sections
        claims = await self._extract_claims(chunks)
        logger.info(f"Extracted {len(claims)} claims to validate")

        if not claims:
            logger.info("No claims extracted, validation complete")
            return {
                'enabled': True,
                'total_claims': 0,
                'claims_supported': 0,
                'claims_unsupported': 0,
                'support_rate': 100.0,
                'critical_issues': [],
                'high_priority_issues': [],
                'summary': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            }

        # Step 2: Validate each claim
        validation_results: list[ValidationResult] = []
        for claim in claims:
            result = await self._validate_claim(claim, chunks)
            validation_results.append(result)

        # Step 3: Generate report
        report = self._generate_report(validation_results)

        return report

    async def _extract_claims(self, chunks: list[Any]) -> list[Claim]:
        """
        Extract technical claims from Features/Introduction sections using LLM.

        Args:
            chunks: Document chunks

        Returns:
            List of extracted claims
        """
        claims: list[Claim] = []
        claim_counter = 0

        for chunk in chunks:
            # Only process chunks from claim-making sections
            section_lower = chunk.section_hierarchy.lower() if hasattr(chunk, 'section_hierarchy') else ''
            if not any(section in section_lower for section in self.claim_sections):
                continue

            # Skip if chunk too short
            if len(chunk.content) < 50:
                continue

            try:
                # Ask LLM to extract claims
                prompt = self._build_claim_extraction_prompt(chunk.content)
                response = await self.llm_client.review_text(
                    chunk.content,
                    context=prompt
                )

                # Parse LLM response to extract claims
                extracted_claims = self._parse_claim_response(
                    response, chunk, claim_counter
                )
                claims.extend(extracted_claims)
                claim_counter += len(extracted_claims)

            except Exception as e:
                logger.warning(f"Failed to extract claims from chunk {chunk.chunk_id}: {e}")
                continue

        return claims

    def _build_claim_extraction_prompt(self, content: str) -> str:
        """Build prompt for LLM to extract claims."""
        return f"""Analyze this technical documentation section and extract all TESTABLE CLAIMS that require supporting evidence.

Focus on claims about:
1. Technical capabilities ("supports USB 2.0", "hardware AES-256", "up to 200 MHz")
2. Specifications ("5V-tolerant I/O", "12-bit ADC", "2 MB Flash")
3. Features ("Wi-Fi connectivity", "CAN FD support", "DMA with 16 channels")
4. Performance ("480 Mbps", "100 MIPS", "1.8V to 5.5V operation")

For EACH claim, identify:
- The exact claim text
- What type of evidence would prove it (timing specs, electrical characteristics, functional description, register map, etc.)
- How critical it is (features need specs, capabilities need descriptions)

Format your response as a JSON array of claims:
[
  {{
    "claim": "USB 2.0 high-speed interface",
    "evidence_needed": ["USB timing specifications", "USB electrical characteristics"],
    "critical": true
  }},
  ...
]

Content to analyze:
{content}"""

    def _parse_claim_response(
        self,
        response: Any,
        chunk: Any,
        start_id: int
    ) -> list[Claim]:
        """Parse LLM response into Claim objects."""
        claims: list[Claim] = []

        try:
            # Try to extract JSON from response
            content = response.content if hasattr(response, 'content') else str(response)

            # Find JSON array in response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                claims_data = json.loads(json_match.group())

                for idx, claim_data in enumerate(claims_data):
                    claim = Claim(
                        claim_id=f"claim_{start_id + idx}",
                        claim_text=claim_data.get('claim', ''),
                        claim_type='feature',  # Could be inferred from evidence_needed
                        section=chunk.section_hierarchy if hasattr(chunk, 'section_hierarchy') else '',
                        page_number=chunk.page_start if hasattr(chunk, 'page_start') else 0,
                        requires_evidence=claim_data.get('evidence_needed', []),
                        confidence=response.confidence if hasattr(response, 'confidence') else 0.8
                    )
                    claims.append(claim)

        except Exception as e:
            logger.warning(f"Failed to parse claim response: {e}")

        return claims

    async def _validate_claim(
        self,
        claim: Claim,
        chunks: list[Any]
    ) -> ValidationResult:
        """
        Validate a single claim by searching for supporting evidence.

        Args:
            claim: Claim to validate
            chunks: All document chunks to search

        Returns:
            Validation result
        """
        # Search for evidence
        evidence_found = await self._find_evidence(claim, chunks)

        # Determine if claim is adequately supported
        is_supported = len(evidence_found) > 0
        confidence = self._calculate_support_confidence(claim, evidence_found)

        # Determine severity
        severity = self._determine_severity(claim, evidence_found)

        # Generate suggestion (with RAG enhancement if available)
        suggestion = await self._generate_suggestion(claim, evidence_found)

        # Identify missing evidence
        missing_evidence = self._identify_missing_evidence(claim, evidence_found)

        return ValidationResult(
            claim=claim,
            evidence_found=evidence_found,
            is_supported=is_supported,
            confidence=confidence,
            severity=severity,
            suggestion=suggestion,
            missing_evidence=missing_evidence
        )

    async def _find_evidence(
        self,
        claim: Claim,
        chunks: list[Any]
    ) -> list[Evidence]:
        """
        Search document for evidence supporting the claim.

        Args:
            claim: Claim to find evidence for
            chunks: All document chunks

        Returns:
            List of evidence found (top 3 most relevant)
        """
        evidence: list[Evidence] = []
        evidence_counter = 0

        # Keywords to search for based on claim
        search_keywords = self._extract_keywords_from_claim(claim.claim_text)

        for chunk in chunks:
            # Skip the chunk where claim was made
            if hasattr(chunk, 'section_hierarchy') and chunk.section_hierarchy == claim.section:
                continue

            # Check if chunk might contain relevant evidence
            relevance_score = self._calculate_relevance(
                claim, chunk, search_keywords
            )

            if relevance_score > 0.3:  # Threshold for potential evidence
                chunk_type = getattr(chunk, 'chunk_type', 'text')
                section = getattr(chunk, 'section_hierarchy', '')
                page = getattr(chunk, 'page_start', 0)

                evidence_item = Evidence(
                    evidence_id=f"evidence_{evidence_counter}",
                    evidence_type=self._classify_evidence_type(chunk_type, section),
                    content=chunk.content[:500],  # Limit size
                    location=f"{section} (p.{page})" if section else f"p.{page}",
                    relevance_score=relevance_score
                )
                evidence.append(evidence_item)
                evidence_counter += 1

        # Sort by relevance
        evidence.sort(key=lambda e: e.relevance_score, reverse=True)

        # Keep top 3 most relevant
        return evidence[:3]

    def _extract_keywords_from_claim(self, claim_text: str) -> set[str]:
        """Extract searchable keywords from claim."""
        keywords: set[str] = set()

        # Technical terms patterns
        patterns = [
            r'USB\s*\d+\.\d+',  # USB 2.0
            r'I[2²]C',  # I2C
            r'SPI',
            r'CAN\s*FD?',
            r'Wi-?Fi',
            r'Bluetooth',
            r'AES-?\d+',  # AES-256
            r'\d+[-\s]bit',  # 12-bit
            r'\d+\s*MHz',  # 200 MHz
            r'\d+\s*Mbps',  # 480 Mbps
        ]

        for pattern in patterns:
            matches = re.findall(pattern, claim_text, re.IGNORECASE)
            keywords.update(matches)

        # Also add significant words (simple approach)
        words = claim_text.split()
        for word in words:
            if len(word) > 4 and word[0].isupper():  # Likely a technical term
                keywords.add(word)

        return keywords

    def _calculate_relevance(
        self,
        claim: Claim,
        chunk: Any,
        keywords: set[str]
    ) -> float:
        """Calculate how relevant a chunk is as evidence for a claim."""
        score = 0.0
        chunk_content_lower = chunk.content.lower()

        # Keyword matching
        for keyword in keywords:
            if keyword.lower() in chunk_content_lower:
                score += 0.3

        # Evidence type matching
        chunk_type = getattr(chunk, 'chunk_type', 'text').lower()
        section_lower = getattr(chunk, 'section_hierarchy', '').lower()

        for required in claim.requires_evidence:
            required_lower = required.lower()
            if required_lower in section_lower or required_lower in chunk_type:
                score += 0.5

        # Table/figure boost (often contain specs)
        if chunk_type in ['table', 'figure']:
            score += 0.2

        return min(score, 1.0)  # Cap at 1.0

    def _classify_evidence_type(self, chunk_type: str, section: str) -> str:
        """Classify what type of evidence a chunk represents."""
        section_lower = section.lower()

        if chunk_type == 'table':
            return 'specification_table'
        elif chunk_type == 'figure':
            return 'diagram'
        elif 'timing' in section_lower:
            return 'timing_specification'
        elif 'electrical' in section_lower or 'characteristics' in section_lower:
            return 'electrical_specification'
        elif 'register' in section_lower:
            return 'register_description'
        else:
            return 'functional_description'

    def _calculate_support_confidence(
        self,
        claim: Claim,
        evidence: list[Evidence]
    ) -> float:
        """Calculate confidence that claim is adequately supported."""
        if not evidence:
            return 0.0

        # Weighted average of relevance scores
        total_score = sum(e.relevance_score for e in evidence)
        avg_score = total_score / len(evidence)

        # Boost if we have multiple pieces of evidence
        multi_evidence_boost = min(len(evidence) * 0.1, 0.3)

        return min(avg_score + multi_evidence_boost, 1.0)

    def _determine_severity(
        self,
        claim: Claim,
        evidence: list[Evidence]
    ) -> str:
        """Determine severity of missing/weak evidence."""
        if not evidence:
            # No evidence found - critical if claim says "supports" something
            if any(word in claim.claim_text.lower() for word in ['support', 'feature', 'include']):
                return 'critical'
            return 'high'

        # Weak evidence
        avg_relevance = sum(e.relevance_score for e in evidence) / len(evidence)
        if avg_relevance < 0.5:
            return 'medium'

        return 'low'

    async def _generate_suggestion(
        self,
        claim: Claim,
        evidence: list[Evidence]
    ) -> str:
        """Generate actionable suggestion for improving documentation."""
        # Base suggestion
        if not evidence:
            evidence_needed = ', '.join(claim.requires_evidence)
            base_suggestion = f"Add {evidence_needed} or remove/clarify claim"
        elif sum(e.relevance_score for e in evidence) / len(evidence) < 0.5:
            base_suggestion = f"Strengthen evidence in {evidence[0].location} with more detailed specifications"
        else:
            return "Evidence found and adequate"

        # Enhance with RAG examples if available
        if self.knowledge_base and hasattr(self.knowledge_base, 'is_available') and self.knowledge_base.is_available():
            try:
                # Retrieve similar approved sections
                similar = await self.knowledge_base.retrieve_similar_sections(
                    query=claim.claim_text,
                    section_type='features',
                    n_results=2,
                    min_quality=0.8
                )

                if similar:
                    examples = []
                    for i, ex in enumerate(similar[:2], 1):
                        doc_id = ex.get('document_id', 'unknown')
                        examples.append(f"Example {i} from {doc_id}: {ex['content'][:150]}...")

                    rag_context = "\n\nSee approved examples:\n" + "\n".join(examples)
                    return base_suggestion + rag_context
            except Exception as e:
                logger.debug(f"RAG enhancement failed: {e}")

        return base_suggestion

    def _identify_missing_evidence(
        self,
        claim: Claim,
        evidence: list[Evidence]
    ) -> list[str]:
        """Identify what evidence is still missing."""
        missing: list[str] = []
        evidence_types_found = set(e.evidence_type for e in evidence)

        for required in claim.requires_evidence:
            # Simple check if this type of evidence was found
            required_lower = required.lower()
            found = any(
                required_lower in etype.lower()
                for etype in evidence_types_found
            )
            if not found:
                missing.append(required)

        return missing

    def _generate_report(
        self,
        validation_results: list[ValidationResult]
    ) -> dict[str, Any]:
        """Generate completeness validation report."""
        # Group by severity
        by_severity: dict[str, list[ValidationResult]] = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }

        for result in validation_results:
            if not result.is_supported or result.confidence < 0.7:
                by_severity[result.severity].append(result)

        # Calculate stats
        total_claims = len(validation_results)
        supported_claims = sum(1 for r in validation_results if r.is_supported)
        support_rate = (supported_claims / total_claims * 100) if total_claims > 0 else 0

        report = {
            'enabled': True,
            'total_claims': total_claims,
            'claims_supported': supported_claims,
            'claims_unsupported': total_claims - supported_claims,
            'support_rate': round(support_rate, 1),
            'critical_issues': [
                {
                    'claim': r.claim.claim_text,
                    'section': r.claim.section,
                    'page': r.claim.page_number,
                    'missing_evidence': r.missing_evidence,
                    'suggestion': r.suggestion,
                    'confidence': round(r.confidence, 2)
                }
                for r in by_severity['critical']
            ],
            'high_priority_issues': [
                {
                    'claim': r.claim.claim_text,
                    'section': r.claim.section,
                    'evidence_found': len(r.evidence_found),
                    'suggestion': r.suggestion
                }
                for r in by_severity['high']
            ],
            'summary': {
                'critical': len(by_severity['critical']),
                'high': len(by_severity['high']),
                'medium': len(by_severity['medium']),
                'low': len(by_severity['low'])
            }
        }

        return report
