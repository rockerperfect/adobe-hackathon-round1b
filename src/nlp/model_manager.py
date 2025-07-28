"""
Model manager - Model loading, caching, and switching utilities for NLP models.

Purpose:
    Provides the ModelManager class for managing transformer, language detection, and other NLP models.
    Supports model loading, caching, availability checks, reload, and switching with thread/process safety.

Structure:
    - ModelManager class: main model management logic

Dependencies:
    - config.model_config (for model paths)
    - sentence_transformers, langdetect, etc. (for models)
    - src.utils.logger

Integration Points:
    - Used by EmbeddingEngine, MultilingualHandler, and other NLP modules
    - Ensures models are loaded and available for downstream use

NOTE: No hardcoded model paths; all config-driven.
"""

from typing import Any, Dict, Optional
import threading
import logging
from config import model_config
from src.utils.logger import get_logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore
try:
    from langdetect import DetectorFactory
except ImportError:
    DetectorFactory = None  # type: ignore


class ModelManager:
    """
    ModelManager manages loading, caching, and switching of NLP models.

    Args:
        config (dict): Project configuration loaded from config files or environment variables.
        logger (logging.Logger, optional): Logger instance for structured logging.

    Attributes:
        config (dict): Loaded configuration.
        logger (logging.Logger): Logger instance.
        _models (dict): Model cache.
        _lock (threading.Lock): Thread lock for safe model access.

    Raises:
        RuntimeError: If required models cannot be loaded.

    Limitations:
        Only manages models specified in config.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the ModelManager.

        Args:
            config (dict, optional): Project configuration. If None, loads from model_config.
            logger (logging.Logger, optional): Logger instance. If None, uses default project logger.
        """
        self.config = config or model_config.load_config()
        self.logger = logger or get_logger(__name__)
        self._models: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def load_model(self, model_name: str) -> Any:
        """
        Load a model by name, using config for path/type.

        Args:
            model_name (str): Name of the model to load (e.g., 'embedding', 'langdetect').

        Returns:
            Any: Loaded model instance.

        Raises:
            RuntimeError: If model cannot be loaded or is unsupported.

        Edge Cases:
            - Returns cached model if already loaded.
            - Raises if model type is unknown.
        """
        with self._lock:
            if model_name in self._models:
                return self._models[model_name]
            if model_name == "embedding":
                if SentenceTransformer is None:
                    self.logger.error("sentence-transformers not installed.")
                    raise RuntimeError("sentence-transformers package is required.")
                model_path = self.config.get("embedding_model_path", model_config.DEFAULT_EMBEDDING_MODEL_PATH)
                model = SentenceTransformer(model_path)
            elif model_name == "langdetect":
                if DetectorFactory is None:
                    self.logger.error("langdetect not installed.")
                    raise RuntimeError("langdetect package is required.")
                model = DetectorFactory()
            else:
                self.logger.error(f"Unknown model name: {model_name}")
                raise RuntimeError(f"Unknown model name: {model_name}")
            self._models[model_name] = model
            self.logger.info(f"Loaded model '{model_name}' successfully.")
            return model

    def get_model(self, model_name: str) -> Any:
        """
        Get a loaded model by name, loading it if necessary.

        Args:
            model_name (str): Name of the model to retrieve.

        Returns:
            Any: Loaded model instance.

        Raises:
            RuntimeError: If model cannot be loaded.
        """
        return self.load_model(model_name)

    def reload_model(self, model_name: str) -> Any:
        """
        Reload a model by name, replacing any cached instance.

        Args:
            model_name (str): Name of the model to reload.

        Returns:
            Any: Reloaded model instance.

        Raises:
            RuntimeError: If model cannot be loaded.
        """
        with self._lock:
            if model_name in self._models:
                del self._models[model_name]
            return self.load_model(model_name)

    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is loaded and available.

        Args:
            model_name (str): Name of the model to check.

        Returns:
            bool: True if model is loaded, False otherwise.
        """
        with self._lock:
            return model_name in self._models

    # TODO: Add support for additional model types and advanced management if needed.
