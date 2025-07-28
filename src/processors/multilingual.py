"""
Multilingual processor - ENHANCED from Round 1A for Round 1B.
OCR and language support with persona-driven multilingual capabilities.

This module provides comprehensive multilingual support for Round 1B including:
- Language detection across 50+ languages
- OCR processing for scanned documents
- Text normalization for different scripts
- Cross-lingual content preprocessing
- Integration with persona-driven analysis

Enhanced for Round 1B:
- Batch processing support for multiple documents
- Integration with embedding models for semantic understanding
- Language-specific text preprocessing
- Support for mixed-language documents
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Import language detection with fallback
try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available - language detection limited")

# Import configuration
from ...config.language_config import (
    SUPPORTED_LANGUAGES,
    PREPROCESSING_RULES,
    NORMALIZATION_PATTERNS,
    LANGUAGE_DETECTION,
    OCR_CONFIG
)


class MultilingualProcessor:
    """
    Enhanced multilingual processor for Round 1B persona-driven analysis.
    
    Provides comprehensive language support including detection, OCR,
    text normalization, and preprocessing for semantic analysis.
    """
    
    def __init__(self):
        """Initialize the multilingual processor."""
        self.logger = logging.getLogger(__name__)
        self.supported_languages = SUPPORTED_LANGUAGES
        self.preprocessing_rules = PREPROCESSING_RULES
        self.normalization_patterns = NORMALIZATION_PATTERNS
        self.language_detection_config = LANGUAGE_DETECTION
        self.ocr_config = OCR_CONFIG
        
        # Initialize OCR if available
        self._initialize_ocr()
    
    def _initialize_ocr(self) -> None:
        """Initialize OCR capabilities with language support."""
        try:
            # Test Tesseract availability
            pytesseract.get_tesseract_version()
            self.ocr_available = True
            self.logger.info("OCR initialized successfully")
        except Exception as e:
            self.ocr_available = False
            self.logger.warning(f"OCR not available: {e}")
    
    def detect_language(self, text: str, fallback: str = "en") -> str:
        """
        Detect the primary language of the given text.
        
        Args:
            text: Text to analyze for language detection
            fallback: Fallback language if detection fails
            
        Returns:
            Language code (e.g., 'en', 'fr', 'ja')
        """
        if not text or len(text.strip()) < self.language_detection_config["min_text_length"]:
            return fallback
        
        if not LANGDETECT_AVAILABLE:
            self.logger.warning("Language detection not available, using fallback")
            return fallback
        
        try:
            # Use sample for detection if text is very long
            sample_text = text[:self.language_detection_config["sample_size"]]
            
            # Get language probabilities
            lang_probs = detect_langs(sample_text)
            
            if lang_probs:
                primary_lang = lang_probs[0]
                
                # Check confidence threshold
                if primary_lang.prob >= self.language_detection_config["confidence_threshold"]:
                    detected_lang = primary_lang.lang
                    
                    # Validate against supported languages
                    if detected_lang in self.supported_languages:
                        self.logger.debug(f"Detected language: {detected_lang} (confidence: {primary_lang.prob:.2f})")
                        return detected_lang
                    else:
                        self.logger.debug(f"Detected unsupported language: {detected_lang}, using fallback")
                        return fallback
                else:
                    self.logger.debug(f"Low confidence detection: {primary_lang.prob:.2f}, using fallback")
                    return fallback
            else:
                return fallback
                
        except Exception as e:
            self.logger.warning(f"Language detection failed: {e}")
            return fallback
    
    def detect_multiple_languages(self, text: str) -> List[Tuple[str, float]]:
        """
        Detect multiple languages in text with confidence scores.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of (language_code, confidence) tuples
        """
        if not LANGDETECT_AVAILABLE or not text:
            return [("en", 1.0)]
        
        try:
            lang_probs = detect_langs(text[:self.language_detection_config["sample_size"]])
            
            # Filter by confidence threshold and supported languages
            results = []
            for lang_prob in lang_probs:
                if (lang_prob.prob >= self.language_detection_config["multi_language_threshold"] and
                    lang_prob.lang in self.supported_languages):
                    results.append((lang_prob.lang, lang_prob.prob))
            
            return results if results else [("en", 1.0)]
            
        except Exception as e:
            self.logger.warning(f"Multi-language detection failed: {e}")
            return [("en", 1.0)]
    
    def preprocess_text(self, text: str, language: str = "en") -> str:
        """
        Apply language-specific text preprocessing.
        
        Args:
            text: Text to preprocess
            language: Language code for specific preprocessing rules
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Determine preprocessing rule set based on language
        rule_set = self._get_rule_set_for_language(language)
        rules = self.preprocessing_rules.get(rule_set, {}).get("rules", {})
        
        processed_text = text
        
        # Apply normalization patterns
        if rules.get("normalize_spacing"):
            processed_text = self._normalize_spacing(processed_text)
        
        if rules.get("handle_mixed_punctuation"):
            processed_text = self._normalize_punctuation(processed_text, language)
        
        if rules.get("character_normalization"):
            processed_text = self._normalize_characters(processed_text, language)
        
        if rules.get("direction_handling") and self.supported_languages.get(language, {}).get("rtl"):
            processed_text = self._handle_rtl_text(processed_text)
        
        return processed_text.strip()
    
    def _get_rule_set_for_language(self, language: str) -> str:
        """Determine which preprocessing rule set to use for a language."""
        lang_info = self.supported_languages.get(language, {})
        
        if lang_info.get("rtl"):
            return "rtl"
        elif lang_info.get("complex_script"):
            if language in ["ja", "zh", "ko"]:
                return "asian"
            else:
                return "complex"
        else:
            return "latin"
    
    def _normalize_spacing(self, text: str) -> str:
        """Normalize whitespace patterns."""
        pattern = self.normalization_patterns["whitespace"]["pattern"]
        replacement = self.normalization_patterns["whitespace"]["replacement"]
        return re.sub(pattern, replacement, text)
    
    def _normalize_punctuation(self, text: str, language: str) -> str:
        """Normalize language-specific punctuation."""
        mixed_punct = self.normalization_patterns["mixed_punctuation"]
        
        if language in ["ja", "zh", "ko"]:
            # Handle Asian punctuation
            for asian_punct, replacement in mixed_punct["asian_periods"].items():
                text = text.replace(asian_punct, replacement)
        elif language in ["ar", "he", "fa"]:
            # Handle Arabic punctuation
            for arabic_punct, replacement in mixed_punct["arabic_punctuation"].items():
                text = text.replace(arabic_punct, replacement)
        
        return text
    
    def _normalize_characters(self, text: str, language: str) -> str:
        """Apply character-level normalization for specific languages."""
        # Basic Unicode normalization
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        
        # Language-specific character handling
        if language in ["ja", "zh", "ko"]:
            # Handle full-width to half-width conversion if needed
            text = self._normalize_asian_characters(text)
        
        return text
    
    def _normalize_asian_characters(self, text: str) -> str:
        """Normalize Asian character encodings."""
        # Convert full-width ASCII to half-width
        result = ""
        for char in text:
            code = ord(char)
            if 0xFF01 <= code <= 0xFF5E:  # Full-width ASCII range
                result += chr(code - 0xFEE0)
            else:
                result += char
        return result
    
    def _handle_rtl_text(self, text: str) -> str:
        """Handle right-to-left text processing."""
        # Basic RTL text handling - preserve direction markers
        # More sophisticated RTL processing can be added as needed
        return text
    
    def process_ocr_image(self, image_path: Path, language: str = "en") -> str:
        """
        Process an image using OCR with language-specific settings.
        
        Args:
            image_path: Path to the image file
            language: Language code for OCR
            
        Returns:
            Extracted text from the image
        """
        if not self.ocr_available:
            self.logger.error("OCR not available")
            return ""
        
        try:
            # Load and preprocess image
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Could not load image: {image_path}")
                return ""
            
            # Apply preprocessing
            processed_image = self._preprocess_image_for_ocr(image)
            
            # Get OCR language code
            tesseract_lang = self._get_tesseract_language_code(language)
            
            # Configure OCR
            config = self._get_ocr_config()
            
            # Perform OCR
            text = pytesseract.image_to_string(
                processed_image,
                lang=tesseract_lang,
                config=config
            )
            
            # Post-process extracted text
            return self.preprocess_text(text, language)
            
        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            return ""
    
    def _preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image to improve OCR accuracy."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply preprocessing based on configuration
        if self.ocr_config["preprocessing"]["resize_factor"] != 1:
            factor = self.ocr_config["preprocessing"]["resize_factor"]
            height, width = gray.shape
            gray = cv2.resize(gray, (int(width * factor), int(height * factor)))
        
        if self.ocr_config["preprocessing"]["noise_removal"]:
            gray = cv2.medianBlur(gray, 3)
        
        if self.ocr_config["preprocessing"]["contrast_enhancement"]:
            gray = cv2.equalizeHist(gray)
        
        return gray
    
    def _get_tesseract_language_code(self, language: str) -> str:
        """Get Tesseract language code for the given language."""
        return self.ocr_config["tesseract_languages"].get(language, "eng")
    
    def _get_ocr_config(self) -> str:
        """Get OCR configuration string for Tesseract."""
        confidence = self.ocr_config["confidence_threshold"]
        return f"--oem 3 --psm 6 -c tessedit_char_whitelist='' -c tessedit_char_blacklist='' -c tessedit_min_orientation_margin={confidence}"
    
    def is_multilingual_document(self, text_blocks: List[Any]) -> bool:
        """
        Determine if a document contains multiple languages.
        
        Args:
            text_blocks: List of text blocks from the document
            
        Returns:
            True if multiple languages are detected
        """
        if len(text_blocks) < 5:  # Need sufficient text for analysis
            return False
        
        # Sample text from different parts of the document
        sample_texts = []
        step = max(1, len(text_blocks) // 10)  # Sample up to 10 blocks
        
        for i in range(0, len(text_blocks), step):
            block = text_blocks[i]
            text = getattr(block, 'text', str(block))
            if len(text.strip()) > 50:  # Sufficient text for language detection
                sample_texts.append(text)
        
        # Detect languages in samples
        detected_languages = set()
        for text in sample_texts[:10]:  # Limit to avoid excessive processing
            lang = self.detect_language(text)
            detected_languages.add(lang)
        
        return len(detected_languages) > 1
    
    def get_language_info(self, language_code: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a language.
        
        Args:
            language_code: Language code
            
        Returns:
            Dictionary with language information
        """
        return self.supported_languages.get(language_code, {
            "name": "Unknown",
            "family": "Unknown",
            "rtl": False,
            "complex_script": False
        })


# Singleton instance for global use
multilingual_processor = MultilingualProcessor()
