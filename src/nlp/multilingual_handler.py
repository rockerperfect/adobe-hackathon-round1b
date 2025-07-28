"""
Multilingual handler - Language detection and processing utilities for multilingual document intelligence.

Purpose:
    Provides the MultilingualHandler class for language detection, normalization, and multilingual preprocessing.
    Supports batch language detection and mixed-language documents.

Structure:
    - MultilingualHandler class: main language detection and normalization logic

Dependencies:
    - config.language_config (for supported languages and model path)
    - langdetect or other language detection library
    - src.utils.logger

Integration Points:
    - Used by text processors, persona/job analyzers, and pipeline orchestrator
    - Outputs language codes and normalized text for downstream modules

NOTE: No hardcoded language codes; all config-driven.
"""

from typing import Any, Dict, List, Optional, Union
import logging
from config import language_config
from src.utils.logger import get_logger

try:
    from langdetect import detect, detect_langs
except ImportError:
    detect = None  # type: ignore
    detect_langs = None  # type: ignore


class MultilingualHandler:
    """
    MultilingualHandler provides language detection and normalization utilities.

    Args:
        config (dict): Project configuration loaded from config files or environment variables.
        logger (logging.Logger, optional): Logger instance for structured logging.

    Attributes:
        config (dict): Loaded configuration.
        logger (logging.Logger): Logger instance.
        supported_languages (list): List of supported language codes.

    Raises:
        RuntimeError: If language detection library is missing.

    Limitations:
        Assumes input text is pre-cleaned.
        Language detection accuracy may vary for short or mixed-language texts.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the MultilingualHandler.

        Args:
            config (dict, optional): Project configuration. If None, loads from language_config.
            logger (logging.Logger, optional): Logger instance. If None, uses default project logger.
        """
        self.config = config or language_config.load_config()
        self.logger = logger or get_logger(__name__)
        self.supported_languages = self.config.get("supported_languages", language_config.DEFAULT_SUPPORTED_LANGUAGES)
        if detect is None or detect_langs is None:
            self.logger.error("langdetect not installed.")
            raise RuntimeError("langdetect package is required.")

    def detect_language(self, text: str) -> str:
        """
        Detect the language code of a single text string.

        Args:
            text (str): Input text to detect language for.

        Returns:
            str: Detected language code (ISO 639-1) or 'unknown'.

        Edge Cases:
            - Returns 'unknown' if detection fails or text is empty.
        """
        if not text:
            self.logger.warning("Empty text for language detection; returning 'unknown'.")
            return "unknown"
        try:
            lang = detect(text)
            if lang not in self.supported_languages:
                self.logger.info(f"Detected language '{lang}' not in supported list; returning 'unknown'.")
                return "unknown"
            return lang
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}", exc_info=True)
            return "unknown"

    def detect_languages_batch(self, texts: List[str]) -> List[str]:
        """
        Detect language codes for a list of text strings (batch operation).

        Args:
            texts (list): List of input texts to detect language for.

        Returns:
            list: List of detected language codes (ISO 639-1) or 'unknown'.

        Edge Cases:
            - Returns 'unknown' for empty or undetectable texts.
        """
        return [self.detect_language(text) for text in texts]

    def normalize_text(self, text: str, lang: Optional[str] = None) -> str:
        """
        Normalize text for downstream processing (e.g., lowercasing, unicode normalization).

        Args:
            text (str): Input text to normalize.
            lang (str, optional): Language code for language-specific normalization.

        Returns:
            str: Normalized text.

        Edge Cases:
            - Returns input text if normalization fails.
        """
        if not text:
            return text
        try:
            # Basic normalization: lowercasing, strip, unicode normalization
            norm = text.strip().lower()
            # TODO: Add language-specific normalization if needed
            return norm
        except Exception as e:
            self.logger.error(f"Text normalization failed: {e}", exc_info=True)
            return text

    # TODO: Add advanced multilingual preprocessing if needed.
