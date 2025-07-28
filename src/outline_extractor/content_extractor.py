"""
Content extractor - NEW for Round 1B.

Purpose:
    Implements the ContentExtractor class for extracting section content from PDF documents based on detected headings.

Structure:
    - ContentExtractor class: main extraction logic

Dependencies:
    - config.settings (for extraction config)
    - src.utils.logger (for logging)

Integration Points:
    - Accepts parsed PDF structure (from PDF parsers), heading information (from heading_detector), and configuration
    - Returns structured representation of extracted sections for downstream modules (e.g., intelligence pipeline)

NOTE: No hardcoded section names, language codes, or document structure assumptions; all config-driven or dynamic.
"""

from typing import Any, Dict, List, Optional
import logging
from config import settings
from src.utils.logger import get_logger


class ContentExtractor:
    """
    ContentExtractor extracts section content from PDF documents based on detected headings.

    Args:
        config (dict): Project configuration loaded from config files or environment variables.
        logger (logging.Logger, optional): Logger instance for structured logging.

    Attributes:
        config (dict): Loaded configuration.
        logger (logging.Logger): Logger instance.

    Raises:
        ValueError: If input is missing or invalid.

    Limitations:
        Assumes input PDF structure and headings are pre-validated.
        Extraction accuracy may vary for ambiguous or poorly structured documents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the ContentExtractor.

        Args:
            config (dict, optional): Project configuration. If None, loads from settings.
            logger (logging.Logger, optional): Logger instance. If None, uses default project logger.
        """
        self.config = config or settings.load_config()
        self.logger = logger or get_logger(__name__)

    def extract_sections(
        self,
        pdf_structure: Dict[str, Any],
        headings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract section content from a parsed PDF structure using detected headings.

        Args:
            pdf_structure (dict): Parsed PDF structure (from PDF parsers).
            headings (list): List of detected heading dicts (from heading_detector).

        Returns:
            list: List of extracted section dicts, each with content and metadata (heading, page numbers, language, etc.).

        Raises:
            ValueError: If input is missing or invalid.

        Edge Cases:
            - Handles overlapping headings, missing headings, ambiguous section boundaries.
            - Returns empty list if no headings or content found.
        """
        if not pdf_structure or not headings:
            self.logger.warning("PDF structure or headings missing; returning empty section list.")
            return []

        # Example structure: pdf_structure['pages'] = [{ 'number': 1, 'text': ... }, ...]
        pages = pdf_structure.get('pages', [])
        if not pages:
            self.logger.warning("No pages found in PDF structure.")
            return []

        # Sort headings by page and position for sequential extraction
        sorted_headings = sorted(headings, key=lambda h: (h.get('page', 0), h.get('position', 0)))
        sections = []
        for idx, heading in enumerate(sorted_headings):
            start_page = heading.get('page', 0)
            start_pos = heading.get('position', 0)
            end_page = sorted_headings[idx + 1].get('page', len(pages)) if idx + 1 < len(sorted_headings) else len(pages)
            end_pos = sorted_headings[idx + 1].get('position', None) if idx + 1 < len(sorted_headings) else None

            # Extract text between current heading and next heading
            section_text = self._extract_text_between(pages, start_page, start_pos, end_page, end_pos)
            section_metadata = {
                'heading': heading.get('text'),
                'start_page': start_page,
                'end_page': end_page,
                'language': heading.get('language', 'unknown'),
                'heading_metadata': heading,
            }
            sections.append({
                'content': section_text,
                'metadata': section_metadata,
            })
        self.logger.info(f"Extracted {len(sections)} sections from PDF.")
        return sections

    def _extract_text_between(
        self,
        pages: List[Dict[str, Any]],
        start_page: int,
        start_pos: int,
        end_page: int,
        end_pos: Optional[int]
    ) -> str:
        """
        Extract text between two positions across pages.

        Args:
            pages (list): List of page dicts.
            start_page (int): Starting page number (0-based).
            start_pos (int): Starting position within the page.
            end_page (int): Ending page number (exclusive, 0-based).
            end_pos (int, optional): Ending position within the ending page.

        Returns:
            str: Extracted text between the specified positions.

        Edge Cases:
            - Handles out-of-bounds indices and missing text.
        """
        content = []
        for i in range(start_page, min(end_page, len(pages))):
            page = pages[i]
            text = page.get('text', '')
            if i == start_page and i == end_page - 1 and end_pos is not None:
                # Single page, bounded by start_pos and end_pos
                content.append(text[start_pos:end_pos])
            elif i == start_page:
                content.append(text[start_pos:])
            elif i == end_page - 1 and end_pos is not None:
                content.append(text[:end_pos])
            else:
                content.append(text)
        return '\n'.join(content)
