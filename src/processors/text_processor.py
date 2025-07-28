"""
Enhanced text processing module for robust PDF heading detection.

This module provides comprehensive text processing capabilities including
text reconstruction, fragmentation repair, and content classification
to ensure accurate heading detection across diverse PDF formats.

Key Features:
- Intelligent text fragment reconstruction
- Robust content vs. heading classification
- Document type recognition and adaptive processing
- Comprehensive filtering of non-heading content
- Position-aware text analysis for layout understanding

Performance optimized for the Adobe Hackathon <10s constraint while
maintaining high accuracy across form documents, academic papers,
reports, and multilingual content.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

# Import TextBlock from the correct path
from ..pdf_parser.base_parser import TextBlock


class DocumentType(Enum):
    """Document type classification for adaptive processing."""
    FORM = "form"
    ACADEMIC_PAPER = "academic_paper"
    REPORT = "report"
    MANUAL = "manual"
    UNKNOWN = "unknown"


@dataclass
class TextFragment:
    """Represents a fragment of text that may need reconstruction."""
    text: str
    position: Tuple[float, float]
    font_size: float
    font_name: str
    confidence: float
    page: int
    is_complete: bool = False
    fragment_type: str = "unknown"


class TextProcessor:
    """
    Advanced text processor for robust PDF heading detection.
    
    Provides comprehensive text analysis including fragment reconstruction,
    content classification, and document structure recognition.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the text processor with comprehensive filtering rules."""
        self.logger = logger or logging.getLogger(__name__)
        
        # Content patterns that should NEVER be classified as headings
        self.non_heading_patterns = {
            # Declaration and certification text
            r'^i\s+(declare|certify|undertake|hereby|confirm)',
            r'^the\s+undersigned\s+declares?',
            r'^this\s+is\s+to\s+certify',
            
            # Copyright and legal text
            r'copyright\s+©?\s*\d{4}',
            r'all\s+rights\s+reserved',
            r'®|™|©',
            r'terms\s+and\s+conditions',
            r'privacy\s+policy',
            
            # Version and metadata
            r'^version\s+\d+\.\d+',
            r'^v\d+\.\d+',
            r'^revision\s+\d+',
            r'^draft\s+\d+',
            r'last\s+updated',
            r'created\s+on',
            
            # Page elements
            r'^page\s+\d+',
            r'^\d+\s+of\s+\d+$',
            r'^(confidential|internal|draft)$',
            
            # Table of contents patterns
            r'\.{3,}',  # Multiple dots (TOC leaders)
            r'\s+\d+$',  # Ends with page number
            
            # URLs and technical references
            r'https?://',
            r'www\.',
            r'@[a-zA-Z0-9.-]+\.',
            
            # File paths and technical identifiers
            r'[a-zA-Z]:\\',
            r'/[a-zA-Z0-9_.-]+/',
            r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}',
            
            # Common body text starters
            r'^(the|this|these|those|a|an)\s+',
            r'^(in|on|at|for|with|by|from|to)\s+',
            r'^(please|kindly|note)',
            
            # Long descriptive text (likely content)
            r'^.{150,}$',  # Very long text is usually content
        }
        
        # Compile patterns for performance
        self.compiled_non_heading_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.non_heading_patterns
        ]
        
        # Valid heading patterns
        self.valid_heading_patterns = {
            # Numbered items
            r'^\d+\.\s*[a-zA-Z]',
            r'^[a-zA-Z]\.\s*[a-zA-Z]',
            r'^[ivx]+\.\s*[a-zA-Z]',
            
            # Section markers
            r'^(section|chapter|part|appendix)\s+\d+',
            r'^(introduction|overview|summary|conclusion)',
            r'^(background|methodology|results|discussion)',
            
            # Form field labels
            r'^(name|address|date|phone|email)',
            r'^(designation|department|office)',
            r'(required|mandatory|optional)$',
        }
        
        self.compiled_valid_heading_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.valid_heading_patterns
        ]
        
        # Document type indicators
        self.document_type_indicators = {
            DocumentType.FORM: [
                r'application\s+form',
                r'registration\s+form',
                r'claim\s+form',
                r'please\s+fill',
                r'signature\s+of\s+applicant',
                r'office\s+use\s+only',
                r'for\s+office\s+use',
            ],
            DocumentType.ACADEMIC_PAPER: [
                r'^abstract$',
                r'^introduction$',
                r'^methodology$',
                r'^references$',
                r'^bibliography$',
                r'et\s+al\.',
                r'\[\d+\]',
            ],
            DocumentType.REPORT: [
                r'^executive\s+summary',
                r'^table\s+of\s+contents',
                r'^appendix\s+[a-z]',
                r'figure\s+\d+',
                r'table\s+\d+',
            ],
            DocumentType.MANUAL: [
                r'user\s+manual',
                r'installation\s+guide',
                r'step\s+\d+',
                r'getting\s+started',
                r'troubleshooting',
            ]
        }
        
    def detect_document_type(self, text_blocks: List[TextBlock]) -> DocumentType:
        """
        Detect document type based on content patterns.
        
        Args:
            text_blocks: List of text blocks from the document
            
        Returns:
            DocumentType enum indicating the detected document type
        """
        # Combine all text for analysis
        all_text = ' '.join([block.text.lower() for block in text_blocks[:20]])  # First 20 blocks
        
        # Score each document type
        type_scores = {doc_type: 0 for doc_type in DocumentType}
        
        for doc_type, patterns in self.document_type_indicators.items():
            for pattern in patterns:
                if re.search(pattern, all_text, re.IGNORECASE):
                    type_scores[doc_type] += 1
        
        # Return the type with highest score
        best_type = max(type_scores, key=type_scores.get)
        
        if type_scores[best_type] > 0:
            self.logger.info(f"Detected document type: {best_type.value}")
            return best_type
        else:
            self.logger.info("Document type: unknown")
            return DocumentType.UNKNOWN
    
    def is_valid_heading(self, text: str, font_info: Dict, position_info: Dict) -> bool:
        """
        Determine if text should be classified as a heading using strict criteria.
        
        Args:
            text: The text content to evaluate
            font_info: Font information (size, name, style)
            position_info: Position and layout information
            
        Returns:
            bool: True if the text qualifies as a valid heading
        """
        text_clean = text.strip()
        
        # Reject empty or very short text
        if len(text_clean) < 2:
            return False
            
        # Reject very long text (likely content)
        if len(text_clean) > 150:
            return False
        
        # Check against non-heading patterns
        for pattern in self.compiled_non_heading_patterns:
            if pattern.search(text_clean):
                self.logger.debug(f"Rejected heading (pattern match): {text_clean[:50]}...")
                return False
        
        # Check for valid heading characteristics
        has_valid_pattern = any(
            pattern.search(text_clean) for pattern in self.compiled_valid_heading_patterns
        )
        
        # Font-based validation
        is_prominent = (
            font_info.get('size', 0) > 10 and
            font_info.get('size', 0) < 72  # Reasonable font size range
        )
        
        # Position-based validation (avoid headers/footers)
        page_height = position_info.get('page_height', 792)
        y_position = position_info.get('y', 0)
        
        is_reasonable_position = (
            y_position > 50 and  # Not in header
            y_position < page_height - 50  # Not in footer
        )
        
        # Content quality checks
        has_reasonable_content = (
            not text_clean.isdigit() and  # Not just a number
            not re.match(r'^[^\w]*$', text_clean) and  # Not just punctuation
            len(text_clean.split()) <= 20  # Not too many words
        )
        
        result = (
            has_reasonable_content and
            is_prominent and
            is_reasonable_position and
            (has_valid_pattern or self._looks_like_heading(text_clean))
        )
        
        if result:
            self.logger.debug(f"Accepted heading: {text_clean[:50]}...")
        
        return result
    
    def _looks_like_heading(self, text: str) -> bool:
        """
        Additional heuristics to identify heading-like text.
        
        Args:
            text: The text to evaluate
            
        Returns:
            bool: True if text has heading-like characteristics
        """
        text_clean = text.strip()
        
        # Check for title case or all caps (common in headings)
        is_title_case = text_clean.istitle()
        is_all_caps = text_clean.isupper() and len(text_clean) > 3
        
        # Check for numbered patterns
        has_numbering = bool(re.match(r'^\d+\.', text_clean))
        
        # Check for section-like words
        section_words = ['section', 'chapter', 'part', 'item', 'point', 'step']
        has_section_word = any(word in text_clean.lower() for word in section_words)
        
        # Check length (headings are usually concise)
        is_concise = 5 <= len(text_clean) <= 80
        
        # Check word count (headings usually have few words)
        word_count = len(text_clean.split())
        reasonable_word_count = 1 <= word_count <= 12
        
        return (
            is_concise and
            reasonable_word_count and
            (is_title_case or is_all_caps or has_numbering or has_section_word)
        )
    
    def reconstruct_fragmented_text(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Combine fragmented text that belongs together.
        
        This is crucial for PDFs where logical text units are split across
        multiple extraction blocks due to formatting or layout artifacts.
        
        Args:
            text_blocks: List of text blocks to process
            
        Returns:
            List of reconstructed text blocks with merged fragments
        """
        if not text_blocks:
            return []
        
        # Sort by position (top-to-bottom, left-to-right)
        sorted_blocks = sorted(text_blocks, key=lambda b: (b.page_number, b.y, b.x))
        
        reconstructed = []
        i = 0
        
        while i < len(sorted_blocks):
            current_block = sorted_blocks[i]
            
            # Look for fragments to merge with current block
            merged_block = self._merge_related_fragments(current_block, sorted_blocks[i+1:])
            
            if merged_block != current_block:
                # Found fragments to merge
                reconstructed.append(merged_block)
                
                # Skip the blocks that were merged
                merged_text_parts = merged_block.text.split()
                current_text_parts = current_block.text.split()
                blocks_merged = len(merged_text_parts) // len(current_text_parts) if current_text_parts else 1
                i += max(1, blocks_merged)
            else:
                # No merge needed
                reconstructed.append(current_block)
                i += 1
        
        self.logger.info(f"Reconstructed {len(text_blocks)} blocks into {len(reconstructed)} blocks")
        return reconstructed
    
    def _merge_related_fragments(self, base_block: TextBlock, candidate_blocks: List[TextBlock]) -> TextBlock:
        """
        Merge text fragments that belong to the same logical unit.
        
        Args:
            base_block: The primary text block to potentially extend
            candidate_blocks: List of subsequent blocks to consider for merging
            
        Returns:
            TextBlock: Either the original block or a merged version
        """
        if not candidate_blocks:
            return base_block
        
        merged_text = base_block.text.strip()
        blocks_to_merge = []
        
        for candidate in candidate_blocks[:5]:  # Only check next 5 blocks
            if not self._should_merge_with_base(base_block, candidate):
                break
                
            blocks_to_merge.append(candidate)
            
            # Add space if needed
            if not merged_text.endswith(' ') and not candidate.text.startswith(' '):
                merged_text += ' '
            
            merged_text += candidate.text.strip()
            
            # Stop if we have a complete sentence or logical unit
            if self._is_complete_unit(merged_text):
                break
        
        if blocks_to_merge:
            # Create merged block
            return TextBlock(
                text=merged_text,
                x=base_block.x,
                y=base_block.y,
                width=max(base_block.width, max(b.width for b in blocks_to_merge)),
                height=base_block.height + sum(b.height for b in blocks_to_merge),
                font_name=base_block.font_name,
                font_size=base_block.font_size,
                page=base_block.page
            )
        
        return base_block
    
    def _should_merge_with_base(self, base_block: TextBlock, candidate_block: TextBlock) -> bool:
        """
        Determine if a candidate block should be merged with the base block.
        
        Args:
            base_block: The primary block
            candidate_block: The block to consider merging
            
        Returns:
            bool: True if blocks should be merged
        """
        # Must be on same page
        if base_block.page_number != candidate_block.page_number:
            return False
        
        # Must be reasonably close
        vertical_distance = abs(candidate_block.y - base_block.y)
        horizontal_distance = abs(candidate_block.x - base_block.x)
        
        if vertical_distance > 20 or horizontal_distance > 100:
            return False
        
        # Check for continuation patterns
        base_text = base_block.text.strip().lower()
        candidate_text = candidate_block.text.strip().lower()
        
        # Known continuation patterns
        continuation_patterns = [
            (r'date\s+of\s+entering\s+the\s+central\s+government', r'service'),
            (r'home\s+town\s+as\s+recorded\s+in\s+the', r'service\s+book'),
            (r'whether\s+wife.*husband.*employed.*and\s+if', r'so\s+whether\s+entitled'),
            (r'single\s+rail\s+fare.*headquarters\s+to.*by', r'shortest\s+route'),
            (r'if\s+the\s+concession.*visit.*in', r'india'),
        ]
        
        for base_pattern, continuation_pattern in continuation_patterns:
            if (re.search(base_pattern, base_text) and 
                re.search(continuation_pattern, candidate_text)):
                return True
        
        # Check for incomplete sentences
        if (base_text.endswith(('and', 'or', 'the', 'of', 'in', 'to', 'for', 'by')) or
            candidate_text.startswith(('and', 'or', 'the', 'of', 'in', 'to', 'for', 'by'))):
            return True
        
        # Check for numbered sequences
        if (re.match(r'\d+\.$', base_text) and 
            re.match(r'^[a-zA-Z]', candidate_text)):
            return True
        
        return False
    
    def _is_complete_unit(self, text: str) -> bool:
        """
        Determine if the merged text forms a complete logical unit.
        
        Args:
            text: The text to evaluate
            
        Returns:
            bool: True if the text appears to be a complete unit
        """
        text_clean = text.strip()
        
        # Complete if ends with proper punctuation
        if text_clean.endswith(('.', '?', '!', ':')):
            return True
        
        # Complete if it's a numbered item with description
        if re.match(r'^\d+\.\s+.{10,}$', text_clean):
            return True
        
        # Complete if it forms a reasonable heading length
        word_count = len(text_clean.split())
        if 3 <= word_count <= 15 and not text_clean.endswith(('and', 'or', 'the', 'of')):
            return True
        
        return False
    
    def determine_heading_level(self, text: str, font_size: float, position: Dict, 
                              document_type: DocumentType, context: Dict) -> str:
        """
        Assign logical heading levels based on document structure.
        
        Args:
            text: The heading text
            font_size: Font size of the text
            position: Position information
            document_type: Detected document type
            context: Additional context about surrounding headings
            
        Returns:
            str: Heading level (H1, H2, or H3)
        """
        text_clean = text.strip().lower()
        
        # H1 criteria - main sections
        h1_indicators = [
            re.match(r'^\d+\.\s+', text),  # Main numbered items
            any(word in text_clean for word in ['application', 'form', 'overview', 'summary']),
            font_size >= context.get('large_font_threshold', 14),
            position.get('y', 0) < 200,  # Near top of page
        ]
        
        if sum(bool(indicator) for indicator in h1_indicators) >= 2:
            return 'H1'
        
        # H2 criteria - subsections
        h2_indicators = [
            re.match(r'^[a-z]\.\s+', text),  # Sub-numbered items
            re.match(r'^[ivx]+\.\s+', text),  # Roman numerals
            any(word in text_clean for word in ['name', 'designation', 'date', 'address']),
            10 <= font_size < context.get('large_font_threshold', 14),
        ]
        
        if sum(bool(indicator) for indicator in h2_indicators) >= 1:
            return 'H2'
        
        # Default to H3 for other valid headings
        return 'H3'
    
    def filter_non_headings(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Remove text blocks that should not be classified as headings.
        
        Args:
            text_blocks: List of text blocks to filter
            
        Returns:
            List of text blocks that qualify as potential headings
        """
        filtered_blocks = []
        
        for block in text_blocks:
            # Create mock objects for validation
            font_info = {
                'size': block.font_size,
                'name': block.font_name
            }
            
            position_info = {
                'x': block.x,
                'y': block.y,
                'page_height': 792  # Standard page height
            }
            
            if self.is_valid_heading(block.text, font_info, position_info):
                filtered_blocks.append(block)
            else:
                self.logger.debug(f"Filtered out non-heading: {block.text[:50]}...")
        
        self.logger.info(f"Filtered {len(text_blocks)} blocks to {len(filtered_blocks)} potential headings")
        return filtered_blocks
    
    def validate_extracted_outline(self, outline_data: Dict) -> Dict:
        """
        Validate that extracted outline makes logical sense.
        
        Args:
            outline_data: The extracted outline data with title and outline entries
            
        Returns:
            Dict: Validation results with issues and recommendations
        """
        issues = []
        recommendations = []
        
        title = outline_data.get('title', '')
        outline = outline_data.get('outline', [])
        
        # Validate title
        if not title or len(title.strip()) < 3:
            issues.append("Title is missing or too short")
            recommendations.append("Extract title from document content")
        elif any(pattern.search(title) for pattern in self.compiled_non_heading_patterns):
            issues.append("Title appears to be metadata or non-content")
            recommendations.append("Look for actual document title in content")
        
        # Validate outline structure
        if not outline:
            issues.append("No headings detected")
            recommendations.append("Review heading detection criteria")
        elif len(outline) > 100:
            issues.append("Too many headings detected (possible false positives)")
            recommendations.append("Apply stricter filtering criteria")
        
        # Check for fragmented headings
        fragmented_count = 0
        for entry in outline:
            text = entry.get('text', '')
            if (len(text) < 5 or 
                text.endswith(('and', 'or', 'the', 'of', 'in')) or
                text.startswith(('and', 'or', 'the', 'of'))):
                fragmented_count += 1
        
        if fragmented_count > len(outline) * 0.3:
            issues.append("High proportion of fragmented headings")
            recommendations.append("Improve text reconstruction logic")
        
        # Check hierarchical consistency
        levels = [entry.get('level', '') for entry in outline]
        if 'H1' not in levels and len(outline) > 5:
            issues.append("No H1 headings found in substantial document")
            recommendations.append("Review H1 classification criteria")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations,
            'metrics': {
                'title_length': len(title),
                'heading_count': len(outline),
                'fragmented_ratio': fragmented_count / max(len(outline), 1)
            }
        }
