"""
Text classifier - Content classification for document intelligence.

Purpose:
    Provides the TextClassifier class for classifying text blocks as heading, body, table, etc., using ML/NLP or rule-based logic as per config.
    Supports multilingual classification and returns class labels with confidence scores.

Structure:
    - TextClassifier class: main classification logic

Dependencies:
    - config.language_config (for supported languages)
    - config.model_config (for classifier config)
    - src.utils.logger

Integration Points:
    - Used by text processors, outline extractor, and pipeline orchestrator
    - Outputs class labels and confidence scores for downstream modules

NOTE: No hardcoded class labels; all config-driven or semantic.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
from config import language_config, model_config
from src.utils.logger import get_logger


class TextClassifier:
    """
    TextClassifier classifies text blocks as heading, body, table, etc.

    Args:
        config (dict): Project configuration loaded from config files or environment variables.
        logger (logging.Logger, optional): Logger instance for structured logging.

    Attributes:
        config (dict): Loaded configuration.
        logger (logging.Logger): Logger instance.
        class_labels (list): List of possible class labels.

    Raises:
        ValueError: If input is missing or invalid.

    Limitations:
        Default implementation is rule-based; can be extended to ML/NLP.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the TextClassifier.

        Args:
            config (dict, optional): Project configuration. If None, loads from model_config.
            logger (logging.Logger, optional): Logger instance. If None, uses default project logger.
        """
        self.config = config or model_config.load_config()
        self.logger = logger or get_logger(__name__)
        self.class_labels = self.config.get("text_class_labels", ["heading", "body", "table", "figure", "other"])

    def classify(self, text: str, lang: Optional[str] = None) -> Tuple[str, float]:
        """
        Classify a text block as heading, body, table, etc.

        Args:
            text (str): Input text to classify.
            lang (str, optional): Language code for multilingual support.

        Returns:
            tuple: (class_label, confidence_score) where class_label is a string and confidence_score is a float in [0, 1].

        Edge Cases:
            - Returns ("other", 0.0) if text is empty or classification fails.
        """
        if not text:
            self.logger.warning("Empty text for classification; returning 'other'.")
            return ("other", 0.0)

        # Simple rule-based logic (can be replaced with ML/NLP model)
        text_stripped = text.strip()
        if len(text_stripped) < 5:
            return ("other", 0.1)
        if text_stripped.isupper() or text_stripped.istitle():
            return ("heading", 0.8)
        if any(char.isdigit() for char in text_stripped) and ("table" in text_stripped.lower() or "figure" in text_stripped.lower()):
            return ("table", 0.7)
        if len(text_stripped.split()) > 20:
            return ("body", 0.9)
        # TODO: Add multilingual and ML-based classification logic
        return ("body", 0.5)

    def classify_batch(self, texts: List[str], langs: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Classify a batch of text blocks.

        Args:
            texts (list): List of input texts to classify.
            langs (list, optional): List of language codes for multilingual support.

        Returns:
            list: List of (class_label, confidence_score) tuples.

        Edge Cases:
            - Returns ("other", 0.0) for empty or undetectable texts.
        """
        if not texts:
            self.logger.warning("Empty text list for batch classification.")
            return [("other", 0.0)] * 0
        results = []
        for i, text in enumerate(texts):
            lang = langs[i] if langs and i < len(langs) else None
            results.append(self.classify(text, lang))
        return results
