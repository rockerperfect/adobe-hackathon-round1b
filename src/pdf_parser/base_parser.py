"""
Abstract base class for PDF parsers in the Adobe India Hackathon PDF Intelligence System.

This module defines the standardized interface for all PDF parsing implementations
used in Round 1A (outline extraction) and Round 1B (persona-driven analysis).
The base class ensures consistency, modularity, and extensibility across different
PDF processing strategies (PyMuPDF primary, PDFMiner fallback).

Key design principles:
- CPU-only execution (no GPU dependencies)
- Offline operation (no internet access required)
- Memory-efficient processing for Docker constraints
- Multilingual support (Japanese, Chinese, Arabic)
- Performance optimization for <10s per 50-page PDF (Round 1A)
- Modular design for Round 1B integration

Integration points:
- Used by outline_extractor for heading detection
- Supports processors for multilingual text handling
- Provides data structures for utils/file_handler operations
- Foundation for Round 1B batch processing capabilities

Performance considerations:
- Lazy loading of large documents
- Efficient memory management for text extraction
- Resource cleanup for file handles
- Caching support for repeated parsing operations
"""

import abc
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class TextBlock:
    """
    Data structure representing a text block with positioning and formatting information.
    
    Used throughout the PDF intelligence pipeline for heading detection,
    text analysis, and multilingual processing.
    
    Attributes:
        text: The extracted text content
        x: X-coordinate position on the page
        y: Y-coordinate position on the page
        width: Width of the text block
        height: Height of the text block
        font_name: Name of the font used
        font_size: Size of the font
        font_flags: Font formatting flags (bold, italic, etc.)
        page_number: Page number where text was found
    """
    
    def __init__(
        self,
        text: str,
        x: float,
        y: float,
        width: float,
        height: float,
        font_name: str,
        font_size: float,
        font_flags: int,
        page_number: int
    ):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.font_name = font_name
        self.font_size = font_size
        self.font_flags = font_flags
        self.page_number = page_number


class BasePDFParser(abc.ABC):
    """
    Abstract base class for PDF parsers in the Adobe Hackathon PDF Intelligence System.
    
    Defines the standardized interface for all PDF parsing implementations,
    ensuring consistency across PyMuPDF, PDFMiner, and future parser additions.
    
    Design principles:
    - CPU-only execution for Docker compatibility
    - Offline operation (no internet dependencies)
    - Memory-efficient processing
    - Multilingual support (Unicode handling)
    - Performance optimization for hackathon constraints
    
    Integration with pipeline:
    - Feeds text blocks to outline_extractor.heading_detector
    - Provides document metadata for processors.text_processor
    - Supports multilingual processing via processors.multilingual
    - Foundation for Round 1B batch processing
    
    Performance targets:
    - Round 1A: <10 seconds per 50-page PDF
    - Memory usage: Efficient within Docker constraints
    - Resource cleanup: Proper file handle management
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize the PDF parser with a document path.
        
        Args:
            pdf_path: Absolute path to the PDF file to be processed
            
        Raises:
            FileNotFoundError: If the PDF file does not exist
            ValueError: If the path is not a valid PDF file
        """
        self.pdf_path = Path(pdf_path)
        self._validate_pdf_path()
        self._document = None
        self._is_loaded = False
    
    def _validate_pdf_path(self) -> None:
        """
        Validate that the PDF path exists and is accessible.
        
        Raises:
            FileNotFoundError: If the PDF file does not exist
            ValueError: If the path is not a valid PDF file
        """
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        if not self.pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"Invalid PDF file extension: {self.pdf_path}")
    
    @abc.abstractmethod
    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse the PDF document and extract all relevant information.
        
        This is the main entry point for PDF processing, combining text extraction,
        positioning information, and metadata retrieval into a unified data structure.
        
        The returned dictionary provides all information needed for:
        - Heading detection algorithms
        - Multilingual text processing
        - Document structure analysis
        - Round 1B persona-driven analysis
        
        Performance requirements:
        - Must complete within time constraints (10s for Round 1A)
        - Memory-efficient processing for large documents
        - Graceful handling of corrupted or malformed PDFs
        
        Args:
            pdf_path: Absolute path to the PDF file to be processed
            
        Returns:
            Dictionary containing:
            - 'text_blocks': List[TextBlock] with positioned text elements
            - 'metadata': Dict with document properties (title, author, etc.)
            - 'page_count': int number of pages
            - 'processing_time': float seconds taken to process
            - 'language': Optional[str] detected primary language
            - 'is_scanned': bool whether document requires OCR
            
        Raises:
            FileNotFoundError: If the PDF file cannot be accessed
            ValueError: If the PDF format is unsupported or corrupted
            MemoryError: If the document is too large for available memory
            RuntimeError: If parsing fails due to internal errors
        """
        pass
    
    @abc.abstractmethod
    def extract_text_with_positions(self) -> List[TextBlock]:
        """
        Extract all text elements with their positioning and formatting information.
        
        This method provides detailed text blocks that are essential for:
        - Rule-based heading detection (font size, position analysis)
        - Hierarchical structure identification
        - Multilingual text processing
        - OCR integration for scanned documents
        
        Each TextBlock contains:
        - Raw text content (Unicode-safe)
        - Precise positioning (x, y coordinates)
        - Formatting information (font, size, style)
        - Page reference for cross-document analysis
        
        Performance considerations:
        - Lazy evaluation for large documents
        - Memory-efficient text block generation
        - Optimal data structures for heading detection algorithms
        
        Returns:
            List of TextBlock objects containing:
            - Positioned text elements from all pages
            - Font and formatting metadata
            - Page number references
            - Coordinate information for layout analysis
            
        Raises:
            RuntimeError: If document is not loaded or parsing fails
            MemoryError: If text extraction exceeds memory limits
            UnicodeDecodeError: If text encoding issues occur
        """
        pass
    
    @abc.abstractmethod
    def get_document_metadata(self) -> Dict[str, Any]:
        """
        Extract document metadata and properties.
        
        Retrieves essential document information for:
        - Title extraction for JSON output formatting
        - Author and creation metadata
        - Document structure analysis
        - Round 1B persona matching and analysis
        
        The metadata supports:
        - Automatic title detection when headings are unclear
        - Document categorization (academic, business, technical)
        - Multilingual document handling
        - Performance optimization through document profiling
        
        Returns:
            Dictionary containing document metadata:
            - 'title': Optional[str] document title from metadata
            - 'author': Optional[str] document author
            - 'subject': Optional[str] document subject/description
            - 'creator': Optional[str] application that created the PDF
            - 'producer': Optional[str] PDF producer software
            - 'creation_date': Optional[str] document creation timestamp
            - 'modification_date': Optional[str] last modification timestamp
            - 'page_count': int total number of pages
            - 'language': Optional[str] document language if specified
            - 'encrypted': bool whether document is password-protected
            - 'version': Optional[str] PDF version number
            
        Raises:
            RuntimeError: If document is not loaded or metadata cannot be extracted
            ValueError: If metadata format is corrupted or unsupported
        """
        pass
    
    def __enter__(self):
        """Context manager entry for resource management."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper resource cleanup."""
        self.close()
    
    @abc.abstractmethod
    def close(self) -> None:
        """
        Clean up resources and close the PDF document.
        
        Ensures proper resource management for:
        - File handle cleanup
        - Memory deallocation
        - Docker container optimization
        - Prevention of resource leaks during batch processing
        """
        pass
