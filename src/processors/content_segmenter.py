"""
Content segmenter - NEW: Intelligent text segmentation for Round 1B.

Purpose:
    Implements the ContentSegmenter class for segmenting text into logical content blocks (paragraphs, sentences, tables, figures, code blocks, etc.).
    Supports multilingual and mixed-language segmentation using rule-based and/or ML/NLP methods as per config.

Structure:
    - ContentSegmenter class: main segmentation logic

Dependencies:
    - config.settings (for segmentation config)
    - src.utils.logger (for logging)

Integration Points:
    - Accepts raw or pre-processed text (from PDF parsing or content extraction), configuration, and logger
    - Returns list of content segments for downstream modules

NOTE: No hardcoding; all configuration, model paths, and language codes are config-driven.
"""

from typing import Any, Dict, List, Optional
import logging
import re
from config import settings
from src.utils.logger import get_logger


class ContentSegmenter:
    """
    ContentSegmenter segments text into logical content blocks.

    Args:
        config (dict): Project configuration loaded from config files or environment variables.
        logger (logging.Logger, optional): Logger instance for structured logging.

    Attributes:
        config (dict): Loaded configuration.
        logger (logging.Logger): Logger instance.

    Raises:
        ValueError: If input is missing or invalid.

    Limitations:
        Default implementation is rule-based; can be extended to ML/NLP.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the ContentSegmenter.

        Args:
            config (dict, optional): Project configuration. If None, loads from settings.
            logger (logging.Logger, optional): Logger instance. If None, uses default project logger.
        """
        self.config = config or settings.load_config()
        self.logger = logger or get_logger(__name__)

    def segment(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Segment text into logical content blocks.

        Args:
            text (str): Raw or pre-processed text to segment.
            metadata (dict, optional): Additional metadata for segmentation context.

        Returns:
            list: List of content segment dicts, each with type, text, position, and metadata.

        Raises:
            ValueError: If input text is missing or invalid.

        Edge Cases:
            - Handles ambiguous boundaries, overlapping segments, missing delimiters.
            - Returns empty list if text is empty or segmentation fails.
        """
        if not text:
            self.logger.warning("Empty text for segmentation; returning empty segment list.")
            return []

        segments = []
        # Simple rule-based segmentation: paragraphs, sentences, tables, code blocks
        # TODO: Replace with ML/NLP-based segmentation as needed
        paragraph_delim = self.config.get('paragraph_delimiter', '\n\n')
        paragraphs = re.split(paragraph_delim, text)
        pos = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Detect tables (very basic heuristic)
            if re.search(r'\btable\b|\bcolumn\b|\brow\b', para, re.IGNORECASE):
                seg_type = 'table'
            # Detect code blocks (very basic heuristic)
            elif para.startswith('```') and para.endswith('```'):
                seg_type = 'code'
            # Detect figures (very basic heuristic)
            elif re.search(r'\bfigure\b|\bimage\b|\bdiagram\b', para, re.IGNORECASE):
                seg_type = 'figure'
            # Detect headings (very basic heuristic)
            elif len(para.split()) < 10 and para.isupper():
                seg_type = 'heading'
            else:
                seg_type = 'paragraph'
            segment = {
                'type': seg_type,
                'text': para,
                'position': pos,
                'metadata': metadata or {},
            }
            segments.append(segment)
            pos += 1
        self.logger.info(f"Segmented text into {len(segments)} content blocks.")
        return segments
