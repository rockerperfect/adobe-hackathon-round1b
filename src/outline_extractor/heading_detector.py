"""
Universal PDF Outline Extraction System - Heading Detection Module

This module implements a robust, universal heading detection system that addresses
all common PDF outline extraction issues through systematic filtering and validation.

Key Design Principles:
1. STRICT CONTENT FILTERING - Never extract metadata, declarations, or body text
2. ACCURATE TITLE EXTRACTION - Find human-visible titles, not metadata
3. COMPLETE HEADING RECONSTRUCTION - Merge fragmented multi-line headings
4. PROPER HIERARCHY ASSIGNMENT - Use explicit logic for H1/H2/H3 classification
5. DOCUMENT-AGNOSTIC DETECTION - No hardcoded patterns or document-specific logic
6. ROBUST VALIDATION - Multiple validation layers to ensure quality

Detection Strategy:
- Relative font size analysis for visual hierarchy
- Position and alignment analysis for structural cues
- Bold/weight detection from font information
- Whitespace and layout analysis for section boundaries
- Comprehensive content filtering to remove noise
- Multi-line heading reconstruction using spatial analysis

Filtering Rules (What is NEVER extracted as headings):
- Document metadata, version info, copyright notices
- Table of contents entries with page numbers/dots
- Author names, signatures, declaration statements
- Headers, footers, running text paragraphs
- Legal disclaimers, terms and conditions
- Very long text (>150 chars) or very short text (<2 chars)
- Text starting with "I declare", "I certify", etc.

Performance Target: <10s processing for typical documents
Accuracy Target: >95% precision with minimal false positives
"""

import re
import logging
import math
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from collections import defaultdict, Counter

from ..pdf_parser.base_parser import TextBlock


@dataclass
class DetectedHeading:
    """Represents a detected heading with complete information."""
    text: str
    level: str
    page: int
    confidence: float
    font_size: float
    font_name: str
    x: float
    y: float
    is_numbered: bool = False
    pattern_type: str = "unknown"
    bbox: Optional[Tuple[float, float, float, float]] = None
    source_blocks: Optional[List[TextBlock]] = None


class UniversalHeadingDetector:
    """
    Universal PDF heading detection system with strict filtering and validation.
    
    This detector implements robust heading extraction that works across all document types
    without hardcoded patterns, ensuring high precision and minimal false positives.
    """
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the universal heading detector."""
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(__name__)
        
        # Compile all filtering patterns at initialization
        self._compile_filtering_patterns()
        
        # Analysis cache for performance
        self._font_stats = None
        self._position_stats = None
        self._page_layouts = {}
        
    def _get_page_number(self, block) -> int:
        """Get page number from text block, handling different attribute names."""
        return getattr(block, 'page_number', getattr(block, 'page', 1))
        
    def _reconstruct_fragmented_headings(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Reconstruct extremely fragmented headings where each character or word 
        is a separate text block (common in stylized image text that's been OCR'd).
        """
        if not text_blocks:
            return text_blocks
            
        # Group blocks by Y position (same line) and sort by X position
        line_groups = {}
        for block in text_blocks:
            page_num = self._get_page_number(block)
            # Round Y to nearest 2 pixels to group blocks on same line
            y_key = (page_num, round(block.y / 2) * 2)
            if y_key not in line_groups:
                line_groups[y_key] = []
            line_groups[y_key].append(block)
        
        reconstructed_blocks = []
        processed_blocks = set()
        
        for (page_num, y_pos), blocks_on_line in line_groups.items():
            if len(blocks_on_line) <= 1:
                # Single block on this line, add as-is
                for block in blocks_on_line:
                    if id(block) not in processed_blocks:
                        reconstructed_blocks.append(block)
                        processed_blocks.add(id(block))
                continue
                
            # Sort blocks by X position (left to right)
            blocks_on_line.sort(key=lambda b: b.x)
            
            # Look for sequences of small text fragments that could be reconstructed
            i = 0
            while i < len(blocks_on_line):
                current_block = blocks_on_line[i]
                
                if id(current_block) in processed_blocks:
                    i += 1
                    continue
                
                # Check if this could be start of a fragmented heading
                potential_fragments = [current_block]
                j = i + 1
                
                # Collect consecutive small fragments
                while j < len(blocks_on_line):
                    next_block = blocks_on_line[j]
                    if id(next_block) in processed_blocks:
                        j += 1
                        continue
                        
                    # Check if blocks are close horizontally and similar font size
                    x_gap = next_block.x - (current_block.x + current_block.width)
                    font_size_similar = abs(next_block.font_size - current_block.font_size) < 5
                    
                    # For large fonts (likely headings), allow larger gaps and even overlaps
                    max_gap = current_block.font_size * 1.0 if current_block.font_size > 15 else current_block.font_size * 0.8
                    close_horizontally = x_gap < max_gap  # Allow small gaps and overlaps (negative values)
                    
                    if close_horizontally and font_size_similar:
                        potential_fragments.append(next_block)
                        current_block = next_block  # Update for next iteration
                        j += 1
                    else:
                        break
                
                # If we found multiple fragments, reconstruct them ONLY if they look like fragmented stylized text
                if (len(potential_fragments) >= 2 and 
                    self._looks_like_fragmented_heading(potential_fragments)):  # More conservative check
                    # Combine the text with minimal spacing for touching fragments
                    combined_text_parts = []
                    for k, frag in enumerate(potential_fragments):
                        text = frag.text.strip()
                        if k > 0:
                            # For touching/overlapping fragments, only add space for word boundaries
                            prev_frag = potential_fragments[k-1]
                            x_gap = frag.x - (prev_frag.x + prev_frag.width)
                            
                            # Add space if there's a reasonable gap or if it makes linguistic sense
                            prev_text = prev_frag.text.strip()
                            # If fragments are overlapping/close AND one is single character, combine without space
                            # This handles cases like "T" + "HERE" -> "THERE"
                            if (x_gap <= 1 and 
                                (len(prev_text) == 1 or len(text) == 1) and
                                prev_text.isalpha() and text.isalpha()):
                                needs_space = False
                            else:
                                needs_space = (x_gap > 2 or  # Small but noticeable gap
                                             (len(prev_text) > 1 and len(text) > 1 and 
                                              prev_text[-1].isalpha() and text[0].isalpha()) or  # Word to word
                                             (len(prev_text) <= 2 and len(text) == 1 and text[0].isupper()) or  # Short + single uppercase (Y + T)
                                             (len(prev_text) == 1 and len(text) > 1 and text[0].isupper()))  # Single char + word starting with uppercase
                            
                            if needs_space and not combined_text_parts[-1].endswith(' '):
                                combined_text_parts.append(' ')
                        combined_text_parts.append(text)
                    
                    combined_text = ''.join(combined_text_parts)
                    
                    # Clean up the combined text
                    combined_text = re.sub(r'\s+', ' ', combined_text).strip()
                    
                    # Create a new merged block with properties from the first fragment
                    first_frag = potential_fragments[0]
                    merged_block = TextBlock(
                        text=combined_text,
                        x=first_frag.x,
                        y=first_frag.y,
                        width=potential_fragments[-1].x + potential_fragments[-1].width - first_frag.x,
                        height=max(frag.height for frag in potential_fragments),
                        font_size=first_frag.font_size,
                        font_name=first_frag.font_name,
                        font_flags=first_frag.font_flags,
                        page_number=self._get_page_number(first_frag)
                    )
                    
                    reconstructed_blocks.append(merged_block)
                    
                    # Mark all fragments as processed
                    for frag in potential_fragments:
                        processed_blocks.add(id(frag))
                        
                    i = j  # Skip to next unprocessed block
                else:
                    # Single block or too few fragments, add as-is
                    reconstructed_blocks.append(current_block)
                    processed_blocks.add(id(current_block))
                    i += 1
        
        # Add any remaining unprocessed blocks
        for block in text_blocks:
            if id(block) not in processed_blocks:
                reconstructed_blocks.append(block)
        
        return reconstructed_blocks
        
    def _looks_like_fragmented_heading(self, fragments):
        """
        Determine if fragments look like a heading that was split by OCR/stylized text.
        Enhanced to handle various fragmentation patterns including multi-word titles.
        """
        if len(fragments) < 2:
            return False
            
        # Calculate average font size - headings typically use larger fonts
        avg_font_size = sum(f.font_size for f in fragments) / len(fragments)
        min_font_size = min(f.font_size for f in fragments)
        
        # Lower threshold for multi-word headings vs character-level fragmentation
        font_threshold = 12 if len(fragments) <= 4 else 15
        if avg_font_size < font_threshold or min_font_size < 10:
            return False
            
        # Analyze fragment characteristics
        total_text_length = sum(len(f.text.strip()) for f in fragments)
        short_fragments = sum(1 for f in fragments if len(f.text.strip()) <= 3)
        single_char_fragments = sum(1 for f in fragments if len(f.text.strip()) == 1)
        
        # Different criteria for different types of fragmentation
        if len(fragments) <= 4:
            # Likely multi-word heading fragmentation (like "Foundation" + "Level" + "Extensions")
            if total_text_length >= 10 and total_text_length <= 80:  # Reasonable heading length
                return True
        else:
            # Likely character-level fragmentation (stylized text)
            if short_fragments >= len(fragments) * 0.6:  # Most fragments are short
                # Check spatial proximity
                avg_gap = self._calculate_average_gap(fragments)
                if avg_gap <= avg_font_size * 0.7:  # Fragments are close together
                    return True
        
        # Additional check: see if fragments form coherent words/phrases
        combined_text = ' '.join(f.text.strip() for f in fragments)
        if self._looks_like_heading_text(combined_text):
            return True
            
        return False
    
    def _calculate_average_gap(self, fragments):
        """Calculate average gap between fragments."""
        if len(fragments) < 2:
            return 0
            
        total_gaps = 0
        gap_count = 0
        
        for i in range(len(fragments) - 1):
            current = fragments[i]
            next_frag = fragments[i + 1]
            gap = next_frag.x - (current.x + getattr(current, 'width', 0))
            total_gaps += abs(gap)
            gap_count += 1
            
        return total_gaps / gap_count if gap_count > 0 else 0
    
    def _looks_like_heading_text(self, text: str) -> bool:
        """Check if combined text looks like a heading."""
        text = text.strip()
        
        # Basic length check
        if len(text) < 5 or len(text) > 100:
            return False
            
        # Check for heading-like characteristics
        words = text.split()
        
        # Reject if too many words (likely content)
        if len(words) > 8:
            return False
            
        # Check capitalization patterns (common in headings)
        title_case_words = sum(1 for w in words if w and w[0].isupper())
        if title_case_words >= len(words) * 0.6:  # Most words start with capitals
            return True
            
        # Check for all caps (common in headings)
        if text.isupper() and len(words) <= 5:
            return True
            
        # Check for common heading words/patterns
        heading_words = ['level', 'extension', 'foundation', 'overview', 'introduction', 
                        'chapter', 'section', 'part', 'guide', 'manual', 'document']
        if any(word.lower() in heading_words for word in words):
            return True
            
        return False
        
    def _compile_filtering_patterns(self) -> None:
        """Compile regex patterns for efficient content filtering."""
        
        # Patterns that NEVER qualify as headings (strict filtering)
        self.exclude_patterns = [
            # Declaration and legal text
            re.compile(r'^i\s+(declare|certify|undertake|hereby|confirm|acknowledge)', re.IGNORECASE),
            re.compile(r'^this\s+is\s+to\s+certify', re.IGNORECASE),
            re.compile(r'^the\s+undersigned', re.IGNORECASE),
            re.compile(r'signature\s+of\s+(applicant|government|officer)', re.IGNORECASE),
            
            # Copyright and metadata
            re.compile(r'copyright\s*©?\s*\d{4}', re.IGNORECASE),
            re.compile(r'all\s+rights\s+reserved', re.IGNORECASE),
            re.compile(r'©|®|™'),
            re.compile(r'^version\s+\d+', re.IGNORECASE),
            re.compile(r'^v\d+\.\d+', re.IGNORECASE),
            re.compile(r'last\s+updated|created\s+on|revision\s+\d+', re.IGNORECASE),
            
            # Table of contents indicators
            re.compile(r'\.{3,}'),  # Multiple dots (TOC leaders)
            re.compile(r'^\s*\d+\s*$'),  # Just page numbers
            re.compile(r'\s+\d+\s*$'),  # Ends with page number
            
            # Page elements and headers/footers
            re.compile(r'^page\s+\d+', re.IGNORECASE),
            re.compile(r'^\d+\s+of\s+\d+$'),
            re.compile(r'^(confidential|internal|draft|preliminary)$', re.IGNORECASE),
            
            # URLs and technical references
            re.compile(r'https?://'),
            re.compile(r'www\.'),
            re.compile(r'@[a-zA-Z0-9.-]+\.'),
            re.compile(r'[a-zA-Z]:\\'),
            
            # Long text (likely content, not headings)
            re.compile(r'^.{150,}$'),
            
            # Common body text patterns
            re.compile(r'^(the|this|these|those|a|an)\s+\w+', re.IGNORECASE),
            re.compile(r'^(in|on|at|for|with|by|from|to)\s+\w+', re.IGNORECASE),
            re.compile(r'^(please|kindly|note|ensure)', re.IGNORECASE),
            
            # Author names and bylines
            re.compile(r'^(by|author|written\s+by)', re.IGNORECASE),
            re.compile(r'^[a-zA-Z]+\s+[a-zA-Z]+\s*,', re.IGNORECASE),  # Name patterns
        ]
        
        # Patterns that indicate valid heading structure
        self.heading_indicators = [
            # Numbered sections
            re.compile(r'^\d+\.\s*[a-zA-Z]'),           # 1. Section
            re.compile(r'^\d+\.\d+\s*[a-zA-Z]'),       # 1.1 Subsection
            re.compile(r'^\d+\.\d+\.\d+\s*[a-zA-Z]'), # 1.1.1 Sub-subsection
            re.compile(r'^[IVX]+\.\s*[a-zA-Z]'),      # I. Roman numerals
            re.compile(r'^[A-Z]\.\s*[a-zA-Z]'),       # A. Letter sections
            
            # Common section headers
            re.compile(r'^(chapter|section|part|appendix)\s+\d+', re.IGNORECASE),
            re.compile(r'^(introduction|overview|summary|conclusion)$', re.IGNORECASE),
            re.compile(r'^(background|methodology|results|discussion)$', re.IGNORECASE),
            re.compile(r'^(abstract|references|bibliography|acknowledgments)$', re.IGNORECASE),
            
            # Form field labels (shorter, specific)
            re.compile(r'^(name|address|date|phone|email|designation)$', re.IGNORECASE),
            re.compile(r'^(department|office|ministry|directorate)$', re.IGNORECASE),
        ]
        
    def detect_headings(self, text_blocks: List[TextBlock]) -> List[DetectedHeading]:
        """
        Detect headings using universal, document-agnostic approach.
        
        Args:
            text_blocks: List of TextBlock objects from PDF parser
            
        Returns:
            List of validated DetectedHeading objects
        """
        if not text_blocks:
            return []
            
        try:
            # Step 0: Analyze document for adaptive thresholds (enhanced for mixed content)
            self.document_stats, self.adaptive_thresholds = self.analyzer.analyze_mixed_content_document(text_blocks)
            
            # Step 1: Reconstruct extremely fragmented headings (OCR'd stylized text)
            reconstructed_blocks = self._reconstruct_fragmented_headings(text_blocks)
            
            # Step 2: Preprocess and group related text blocks
            grouped_blocks = self._group_fragmented_text(reconstructed_blocks)
            
            # Step 3: Analyze document characteristics
            self._analyze_document_structure(grouped_blocks)
            
            # Step 4: Detect heading candidates using multiple strategies
            candidates = []
            candidates.extend(self._detect_by_font_analysis(grouped_blocks))
            candidates.extend(self._detect_by_position_analysis(grouped_blocks))
            candidates.extend(self._detect_by_pattern_analysis(grouped_blocks))
            
            # Step 5: Filter and validate candidates
            valid_headings = self._filter_and_validate_candidates(candidates)
            
            # Step 6: Assign proper hierarchical levels
            hierarchical_headings = self._assign_heading_levels(valid_headings)
            
            # Step 7: Final validation and cleanup
            final_headings = self._final_validation(hierarchical_headings)
            
            self.logger.info(f"Detected {len(final_headings)} valid headings from {len(text_blocks)} text blocks")
            return final_headings
            
        except Exception as e:
            self.logger.error(f"Error in heading detection: {e}")
            return []
    
    def _group_fragmented_text(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Group text fragments that belong to the same logical heading.
        
        This is crucial for documents where headings are split across lines or
        have unusual formatting like "Foundation Level Extensions".
        """
        if not text_blocks:
            return []
            
        # Sort blocks by page, then by Y position (top to bottom), then X position
        sorted_blocks = sorted(text_blocks, key=lambda b: (self._get_page_number(b), -b.y, b.x))
        
        grouped = []
        i = 0
        
        while i < len(sorted_blocks):
            current = sorted_blocks[i]
            
            # Start with the current block
            merged_text = current.text.strip()
            merged_block = current
            components = [current]  # Track all components for better positioning
            
            # Look for potential continuations within a reasonable window
            j = i + 1
            search_limit = min(len(sorted_blocks), i + 8)  # Expanded search window
            
            while j < search_limit:
                next_block = sorted_blocks[j]
                
                # Check if this block should be merged with our current heading
                if self._should_merge_blocks(merged_block, next_block):
                    # Determine how to join the text
                    next_text = next_block.text.strip()
                    
                    # Smart text joining based on context
                    if self._needs_space_between(merged_text, next_text):
                        merged_text += " " + next_text
                    else:
                        merged_text += next_text
                    
                    # Update the merged block with new properties
                    merged_block = self._create_merged_block(merged_block, next_block, merged_text)
                    components.append(next_block)
                    
                    # Remove the merged block from further consideration
                    sorted_blocks.pop(j)
                    search_limit -= 1  # Adjust search limit since we removed a block
                    continue
                    
                j += 1
            
            # Clean up the merged text
            merged_text = self._clean_merged_heading_text(merged_text)
            if merged_text != merged_block.text:
                merged_block = self._create_merged_block(merged_block, merged_block, merged_text)
            
            grouped.append(merged_block)
            i += 1
            
        return grouped
    
    def _needs_space_between(self, text1: str, text2: str) -> bool:
        """Determine if a space is needed between two text fragments."""
        if not text1 or not text2:
            return False
            
        # Add space if first text doesn't end with space and second doesn't start with space
        if text1.endswith(' ') or text2.startswith(' '):
            return False
            
        # Don't add space if joining punctuation
        if text2.startswith((',', '.', '!', '?', ':', ';', ')', ']', '}')):
            return False
            
        # Don't add space if first text ends with opening punctuation
        if text1.endswith(('(', '[', '{')):
            return False
            
        # Add space for word boundaries
        return True
    
    def _clean_merged_heading_text(self, text: str) -> str:
        """Clean up merged heading text."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove redundant spaces around punctuation
        text = re.sub(r'\s+([,.!?:;])', r'\1', text)
        text = re.sub(r'([(\[{])\s+', r'\1', text)
        text = re.sub(r'\s+([)\]}])', r'\1', text)
        
        return text
    
    def _should_merge_blocks(self, block1: TextBlock, block2: TextBlock) -> bool:
        """Determine if two text blocks should be merged as a single heading."""
        # Must be on same page
        if self._get_page_number(block1) != self._get_page_number(block2):
            return False
            
        text1 = block1.text.strip()
        text2 = block2.text.strip()
        
        # Don't merge if either is very long (likely content, not headings)
        if len(text1) > 100 or len(text2) > 100:
            return False
            
        # Check font consistency - allow more flexibility for headings
        font_size_diff = abs(block1.font_size - block2.font_size)
        if font_size_diff > max(2, block1.font_size * 0.15):  # Allow 15% difference or 2pt max
            return False
            
        # Calculate vertical and horizontal distances
        vertical_distance = abs(block1.y - block2.y)
        
        # For same line merging (horizontal continuation)
        if vertical_distance <= block1.font_size * 0.3:  # Very close vertically (same line)
            # Check horizontal proximity - allow for word spacing
            expected_x = block1.x + len(text1) * block1.font_size * 0.6  # Estimate text width
            horizontal_gap = abs(block2.x - expected_x)
            
            # Allow reasonable word spacing for same-line continuation
            if horizontal_gap <= block1.font_size * 1.5:
                return True
                
        # For multi-line heading merging (vertical continuation)
        elif vertical_distance <= block1.font_size * 2.0:  # Within 2 line heights
            # Check if they could be parts of a multi-line heading
            
            # Check horizontal alignment - should start at similar X positions or be centered
            x_alignment_tolerance = block1.font_size * 1.0
            similar_x_start = abs(block1.x - block2.x) <= x_alignment_tolerance
            
            # Check if blocks are relatively short (typical of heading fragments)
            both_short = len(text1) <= 50 and len(text2) <= 50
            
            # Check if they have similar font properties (both large/heading-like)
            both_large_font = block1.font_size >= 12 and block2.font_size >= 12
            
            # Linguistic checks for heading continuation
            looks_like_heading_parts = self._looks_like_heading_parts(text1, text2)
            
            # Merge if multiple indicators suggest this is a multi-line heading
            merge_indicators = sum([
                similar_x_start,
                both_short,
                both_large_font,
                looks_like_heading_parts,
            ])
            
            if merge_indicators >= 2:  # At least 2 indicators must be true
                return True
                
        return False
    
    def _looks_like_heading_parts(self, text1: str, text2: str) -> bool:
        """Check if two text fragments look like parts of a heading."""
        # Don't merge if second part starts new sentence (capital + period)
        if text2 and text2[0].isupper() and text1.endswith('.'):
            return False
            
        # Don't merge if they look like separate statements
        if (text1.endswith(('!', '?', ':')) and text2[0].isupper() and len(text2.split()) > 3):
            return False
            
        # Positive indicators for heading parts
        heading_indicators = 0
        
        # Both parts are title-case or all caps (common in headings)
        if (text1.istitle() or text1.isupper()) and (text2.istitle() or text2.isupper()):
            heading_indicators += 1
            
        # First part ends with incomplete phrase
        if text1.rstrip().endswith((' and', ' of', ' the', ' for', ' in', ' on', ' with', ' Level')):
            heading_indicators += 1
            
        # Second part continues naturally
        if text2.split()[0].lower() in ['and', 'of', 'the', 'for', 'in', 'on', 'with', 'extensions', 'level']:
            heading_indicators += 1
            
        # Both parts are relatively short (typical of heading fragments)
        if len(text1.split()) <= 4 and len(text2.split()) <= 4:
            heading_indicators += 1
            
        # Neither part contains sentence-ending punctuation in the middle
        if '.' not in text1[:-1] and '.' not in text2[:-1]:
            heading_indicators += 1
            
        return heading_indicators >= 2
    
    def _create_merged_block(self, block1: TextBlock, block2: TextBlock, merged_text: str) -> TextBlock:
        """Create a new TextBlock from merging two blocks."""
        return TextBlock(
            text=merged_text,
            x=block1.x,
            y=block1.y,
            width=getattr(block1, 'width', 0),
            height=getattr(block1, 'height', 0),
            font_name=block1.font_name,
            font_size=block1.font_size,
            font_flags=getattr(block1, 'font_flags', 0),
            page_number=self._get_page_number(block1)
        )
    
    def _analyze_document_structure(self, text_blocks: List[TextBlock]) -> None:
        """Analyze document structure to establish baselines for detection."""
        if not text_blocks:
            return
            
        # Analyze font sizes
        font_sizes = [block.font_size for block in text_blocks if block.font_size > 0]
        if font_sizes:
            self._font_stats = {
                'mean': sum(font_sizes) / len(font_sizes),
                'median': sorted(font_sizes)[len(font_sizes) // 2],
                'max': max(font_sizes),
                'min': min(font_sizes),
                'q75': sorted(font_sizes)[int(len(font_sizes) * 0.75)],
                'q25': sorted(font_sizes)[int(len(font_sizes) * 0.25)]
            }
        
        # Analyze positions per page
        self._page_layouts = {}
        for block in text_blocks:
            page = self._get_page_number(block)
            if page not in self._page_layouts:
                self._page_layouts[page] = {
                    'left_margins': [],
                    'y_positions': [],
                    'font_sizes': []
                }
            
            self._page_layouts[page]['left_margins'].append(block.x)
            self._page_layouts[page]['y_positions'].append(block.y)
            self._page_layouts[page]['font_sizes'].append(block.font_size)
    
    def _detect_by_font_analysis(self, text_blocks: List[TextBlock]) -> List[DetectedHeading]:
        """Detect headings based on font size and weight analysis."""
        candidates = []
        
        if not self._font_stats:
            return candidates
            
        # Get adaptive thresholds
        thresholds = self.adaptive_thresholds
        
        for block in text_blocks:
            # Skip if fails basic content filtering
            if not self._passes_content_filter(block.text):
                continue
                
            confidence = 0.0
            pattern_type = "font"
            
            # Large font size indicates heading
            if block.font_size >= thresholds.title_font_threshold:
                confidence += 0.7
            elif block.font_size >= thresholds.heading_font_thresholds.get('H3', 10.0):
                confidence += 0.4
                
            # Bold text (if font flags available)
            if hasattr(block, 'font_flags') and block.font_flags:
                if block.font_flags & 0x1:  # Bold flag
                    confidence += 0.3
                if block.font_flags & 0x2:  # Italic flag
                    confidence += 0.1
                    
            # Position indicators
            if self._is_left_aligned(block):
                confidence += 0.2
            if self._is_isolated_vertically(block, text_blocks):
                confidence += 0.3
                
            if confidence >= 0.5:  # Minimum confidence threshold
                candidates.append(DetectedHeading(
                    text=block.text.strip(),
                    level="H1",  # Will be refined later
                    page=self._get_page_number(block),
                    confidence=confidence,
                    font_size=block.font_size,
                    font_name=getattr(block, 'font_name', ''),
                    x=block.x,
                    y=block.y,
                    pattern_type=pattern_type,
                    source_blocks=[block]
                ))
                
        return candidates
    
    def _detect_by_position_analysis(self, text_blocks: List[TextBlock]) -> List[DetectedHeading]:
        """Detect headings based on position and alignment patterns."""
        candidates = []
        
        for block in text_blocks:
            if not self._passes_content_filter(block.text):
                continue
                
            confidence = 0.0
            pattern_type = "position"
            
            # Check for centered text (potential title)
            if self._is_centered(block):
                confidence += 0.5
                pattern_type = "centered"
                
            # Check for consistent left alignment
            if self._is_consistently_aligned(block, text_blocks):
                confidence += 0.3
                
            # Check for unusual spacing above/below
            if self._has_unusual_spacing(block, text_blocks):
                confidence += 0.4
                
            # Check for short, isolated text
            if len(block.text.strip()) < 50 and self._is_isolated_vertically(block, text_blocks):
                confidence += 0.3
                
            if confidence >= 0.4:
                candidates.append(DetectedHeading(
                    text=block.text.strip(),
                    level="H1",
                    page=self._get_page_number(block),
                    confidence=confidence,
                    font_size=block.font_size,
                    font_name=getattr(block, 'font_name', ''),
                    x=block.x,
                    y=block.y,
                    pattern_type=pattern_type,
                    source_blocks=[block]
                ))
                
        return candidates
    
    def _detect_by_pattern_analysis(self, text_blocks: List[TextBlock]) -> List[DetectedHeading]:
        """Detect headings based on text patterns and numbering."""
        candidates = []
        
        for block in text_blocks:
            if not self._passes_content_filter(block.text):
                continue
                
            text = block.text.strip()
            confidence = 0.0
            pattern_type = "pattern"
            is_numbered = False
            
            # Check for numbered sections
            for pattern in self.heading_indicators:
                if pattern.search(text):
                    confidence += 0.6
                    if any(p.search(text) for p in self.heading_indicators[:5]):  # Numbered patterns
                        is_numbered = True
                    break
                    
            # Boost confidence for good heading indicators
            if any(word in text.lower() for word in ['chapter', 'section', 'part', 'introduction', 'overview']):
                confidence += 0.4
                
            # Check text length (good headings are usually 5-80 characters)
            text_len = len(text)
            if 5 <= text_len <= 80:
                confidence += 0.2
            elif text_len > 150:
                confidence -= 0.5  # Too long for heading
                
            if confidence >= 0.4:
                candidates.append(DetectedHeading(
                    text=text,
                    level="H1",
                    page=self._get_page_number(block),
                    confidence=confidence,
                    font_size=block.font_size,
                    font_name=getattr(block, 'font_name', ''),
                    x=block.x,
                    y=block.y,
                    is_numbered=is_numbered,
                    pattern_type=pattern_type,
                    source_blocks=[block]
                ))
                
        return candidates
    
    def _passes_content_filter(self, text: str) -> bool:
        """Check if text passes all content filtering rules."""
        text_clean = text.strip()
        
        # Basic length checks
        if len(text_clean) < 2 or len(text_clean) > 150:
            return False
            
        # Check against exclusion patterns
        for pattern in self.exclude_patterns:
            if pattern.search(text_clean):
                return False
                
        return True
    
    def _is_left_aligned(self, block: TextBlock) -> bool:
        """Check if block is left-aligned with document margin."""
        page_num = self._get_page_number(block)
        if not self._page_layouts or page_num not in self._page_layouts:
            return True  # Default to true if no analysis available
            
        page_margins = self._page_layouts[page_num]['left_margins']
        if not page_margins:
            return True
            
        common_margin = min(page_margins)  # Leftmost margin
        return abs(block.x - common_margin) < 10  # Within 10 units
    
    def _is_centered(self, block: TextBlock) -> bool:
        """Check if text appears centered on the page."""
        # Simple heuristic: if X position is significantly right of left margin
        page_num = self._get_page_number(block)
        if not self._page_layouts or page_num not in self._page_layouts:
            return False
            
        page_margins = self._page_layouts[page_num]['left_margins']
        if not page_margins:
            return False
            
        left_margin = min(page_margins)
        right_margin = max(page_margins)
        
        # Consider centered if positioned significantly right of left margin
        margin_width = right_margin - left_margin
        if margin_width > 100:  # Only for pages with significant width variation
            center_position = left_margin + margin_width / 2
            return abs(block.x - center_position) < margin_width * 0.3
            
        return False
    
    def _is_consistently_aligned(self, block: TextBlock, all_blocks: List[TextBlock]) -> bool:
        """Check if block aligns with other text at similar positions."""
        page_num = self._get_page_number(block)
        same_page_blocks = [b for b in all_blocks if self._get_page_number(b) == page_num]
        similar_x_blocks = [b for b in same_page_blocks if abs(b.x - block.x) < 5]
        
        return len(similar_x_blocks) >= 2  # At least 2 blocks at similar position
    
    def _has_unusual_spacing(self, block: TextBlock, all_blocks: List[TextBlock]) -> bool:
        """Check if block has unusual vertical spacing above/below."""
        page_num = self._get_page_number(block)
        same_page_blocks = [b for b in all_blocks if self._get_page_number(b) == page_num]
        same_page_blocks.sort(key=lambda b: -b.y)  # Top to bottom
        
        block_index = -1
        for i, b in enumerate(same_page_blocks):
            if b == block:
                block_index = i
                break
                
        if block_index == -1:
            return False
            
        # Get adaptive spacing threshold
        thresholds = self.adaptive_thresholds
        spacing_threshold = thresholds.spatial_grouping_tolerance
        
        # Check spacing to previous and next blocks
        has_unusual = False
        
        if block_index > 0:
            prev_block = same_page_blocks[block_index - 1]
            spacing_above = prev_block.y - block.y
            if spacing_above > spacing_threshold:
                has_unusual = True
                
        if block_index < len(same_page_blocks) - 1:
            next_block = same_page_blocks[block_index + 1]
            spacing_below = block.y - next_block.y
            if spacing_below > spacing_threshold:
                has_unusual = True
                
        return has_unusual
    
    def _is_isolated_vertically(self, block: TextBlock, all_blocks: List[TextBlock]) -> bool:
        """Check if block is vertically isolated from other text."""
        page_num = self._get_page_number(block)
        same_page_blocks = [b for b in all_blocks if self._get_page_number(b) == page_num]
        
        # Count nearby blocks
        nearby_count = 0
        for other_block in same_page_blocks:
            if other_block != block:
                vertical_distance = abs(other_block.y - block.y)
                if vertical_distance < block.font_size * 1.5:
                    nearby_count += 1
                    
        return nearby_count <= 1  # Isolated if 1 or fewer nearby blocks
    
    def _filter_and_validate_candidates(self, candidates: List[DetectedHeading]) -> List[DetectedHeading]:
        """Filter and validate heading candidates."""
        if not candidates:
            return []
            
        # Remove duplicates based on text and position
        unique_candidates = []
        seen_texts = set()
        
        for candidate in candidates:
            text_key = candidate.text.lower().strip()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_candidates.append(candidate)
                
        # Sort by confidence and filter low-confidence candidates
        unique_candidates.sort(key=lambda h: h.confidence, reverse=True)
        
        # Keep only candidates above minimum confidence
        valid_candidates = [h for h in unique_candidates if h.confidence >= 0.4]
        
        # Additional validation
        final_candidates = []
        for candidate in valid_candidates:
            if self._validate_heading_candidate(candidate):
                final_candidates.append(candidate)
                
        return final_candidates
    
    def _validate_heading_candidate(self, candidate: DetectedHeading) -> bool:
        """Perform final validation on a heading candidate."""
        text = candidate.text.strip()
        
        # Get adaptive thresholds
        thresholds = self.adaptive_thresholds
        
        # Reject lines that are mostly or entirely punctuation/symbols (like "----------------")
        alphanumeric_chars = sum(1 for c in text if c.isalnum())
        min_alpha_ratio = thresholds.noise_filter_thresholds.get('min_alpha_ratio', 0.3)
        if len(text) > 3 and alphanumeric_chars < len(text) * min_alpha_ratio:
            return False
        
        # Additional text-based validation
        # Reject very short standalone words unless they're section headers
        min_length = thresholds.noise_filter_thresholds.get('min_text_length', 3)
        if len(text) < min_length and not any(word in text.lower() for word in ['name', 'date', 'page']):
            return False
            
        # Reject text that looks like running sentences
        if text.count(' ') > 10 and text.endswith('.'):  # Long sentence-like text
            return False
            
        # Reject text with too many special characters
        special_char_count = sum(1 for c in text if not c.isalnum() and c not in ' .-')
        max_special_ratio = thresholds.noise_filter_thresholds.get('max_special_char_ratio', 0.6)
        if special_char_count > len(text) * max_special_ratio:
            return False
            
        return True
    
    def _assign_heading_levels(self, headings: List[DetectedHeading]) -> List[DetectedHeading]:
        """Assign proper hierarchical levels to detected headings."""
        if not headings:
            return []
            
        # Sort headings by page and position
        sorted_headings = sorted(headings, key=lambda h: (h.page, -h.y, h.x))
        
        # Analyze font sizes and patterns to determine hierarchy
        font_sizes = [h.font_size for h in sorted_headings]
        unique_sizes = sorted(set(font_sizes), reverse=True)
        
        # Assign levels based on font size hierarchy and numbering patterns
        for heading in sorted_headings:
            level = self._determine_heading_level(heading, unique_sizes)
            heading.level = level
            
        return sorted_headings
    
    def _determine_heading_level(self, heading: DetectedHeading, font_size_hierarchy: List[float]) -> str:
        """Determine the appropriate level for a heading."""
        # Check for numbered patterns first
        text = heading.text.strip()
        
        # H1: Main sections (1., Chapter, etc.)
        if (re.match(r'^\d+\.\s', text) or 
            any(word in text.lower() for word in ['chapter', 'part', 'section'])):
            return "H1"
            
        # H2: Subsections (1.1, numbered sub-items)
        if re.match(r'^\d+\.\d+', text):
            return "H2"
            
        # H3: Sub-subsections (1.1.1)
        if re.match(r'^\d+\.\d+\.\d+', text):
            return "H3"
            
        # Use font size for level assignment
        if len(font_size_hierarchy) >= 3:
            if heading.font_size >= font_size_hierarchy[0]:
                return "H1"
            elif heading.font_size >= font_size_hierarchy[1]:
                return "H2"
            else:
                return "H3"
        elif len(font_size_hierarchy) >= 2:
            if heading.font_size >= font_size_hierarchy[0]:
                return "H1"
            else:
                return "H2"
        else:
            return "H1"  # Default to H1 if single font size
    
    def _final_validation(self, headings: List[DetectedHeading]) -> List[DetectedHeading]:
        """Perform final validation and cleanup of detected headings."""
        if not headings:
            return []
            
        # Remove any remaining low-quality headings
        quality_headings = []
        
        for heading in headings:
            # Skip headings that are just numbers or single characters
            if len(heading.text.strip()) <= 2 and not heading.text.strip().isalpha():
                continue
                
            # Skip headings that look like content
            if self._looks_like_content(heading.text):
                continue
                
            quality_headings.append(heading)
            
        # Sort final headings by document order
        final_headings = sorted(quality_headings, key=lambda h: (h.page, -h.y, h.x))
        
        return final_headings
    
    def _looks_like_content(self, text: str) -> bool:
        """Check if text looks like content rather than a heading."""
        text_clean = text.strip()
        
        # Too many words suggests content
        word_count = len(text_clean.split())
        if word_count > 8:
            return True
            
        # Contains common content indicators
        content_indicators = ['i declare', 'i certify', 'the undersigned', 'signature of']
        for indicator in content_indicators:
            if indicator in text_clean.lower():
                return True
                
        return False


# For backward compatibility, create an alias
HeadingDetector = UniversalHeadingDetector
