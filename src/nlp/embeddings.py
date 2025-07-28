"""
Embeddings - Sentence Transformers integration for multilingual document intelligence.

Purpose:
    Provides the EmbeddingEngine class for generating multilingual text embeddings using a transformer model.
    Supports batch operations and embedding caching for performance.

Structure:
    - EmbeddingEngine class: main embedding logic
    - Utility functions for embedding and cache management

Dependencies:
    - config.model_config (for model path)
    - sentence_transformers (for transformer model)
    - src.utils.logger

Integration Points:
    - Used by PersonaAnalyzer, SectionRanker, SubsectionExtractor, etc.
    - Outputs embeddings for downstream semantic analysis

NOTE: No hardcoded model paths; all config-driven.
"""

from typing import Any, Dict, List, Optional, Union
import logging
import threading
from config import model_config
from src.utils.logger import get_logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


class EmbeddingEngine:
    """
    EmbeddingEngine loads a multilingual transformer model and generates embeddings for text.

    Args:
        config (dict): Project configuration loaded from config files or environment variables.
        logger (logging.Logger, optional): Logger instance for structured logging.

    Attributes:
        model (SentenceTransformer): Loaded transformer model.
        config (dict): Loaded configuration.
        logger (logging.Logger): Logger instance.
        _cache (dict): Embedding cache for performance.
        _lock (threading.Lock): Thread lock for safe model access.

    Raises:
        RuntimeError: If model loading fails.

    Limitations:
        Assumes model supports all required languages.
        Caching is in-memory only.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the EmbeddingEngine.

        Args:
            config (dict, optional): Project configuration. If None, loads from model_config.
            logger (logging.Logger, optional): Logger instance. If None, uses default project logger.
        """
        self.config = config or model_config.load_config()
        self.logger = logger or get_logger(__name__)
        self.model_path = self.config.get("embedding_model_path", model_config.DEFAULT_EMBEDDING_MODEL_PATH)
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self.model = self._load_model()

    def _load_model(self) -> Any:
        """
        Load the transformer model from the configured path - SPEED OPTIMIZED with singleton.

        Returns:
            SentenceTransformer: Loaded model instance.

        Raises:
            RuntimeError: If model loading fails or dependency missing.
        """
        if SentenceTransformer is None:
            self.logger.error("sentence-transformers not installed.")
            raise RuntimeError("sentence-transformers package is required.")
        
        try:
            # Use singleton pattern to avoid reloading the same model multiple times
            from .model_singleton import ModelSingleton
            singleton = ModelSingleton()
            model = singleton.get_model(self.model_path)
            
            self.logger.info("Embedding model loaded successfully.")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load embedding model: {e}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text string - SPEED OPTIMIZED.

        Args:
            text (str): Input text to embed.

        Returns:
            list: Embedding vector.

        Raises:
            Exception: If embedding fails.

        Edge Cases:
            - Returns cached embedding if available.
            - Handles empty or None input (returns zero vector).
        """
        if not text:
            self.logger.warning("Empty text provided for embedding; returning zero vector.")
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        # QUALITY RESTORATION: Balanced text truncation for quality
        if len(text) > 512:  # Restored for quality
            text = text[:512]
        
        cache_key = f"text:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        with self._lock:
            try:
                # SPEED OPTIMIZATION: Disable progress bar and use batch_size=1
                emb = self.model.encode(
                    text, 
                    show_progress_bar=False,
                    batch_size=1,
                    convert_to_tensor=False
                ).tolist()
                
                # AGGRESSIVE CACHING
                if len(self._cache) < 5000:  # Increase cache size
                    self._cache[cache_key] = emb
                return emb
            except Exception as e:
                self.logger.error(f"Embedding failed: {e}", exc_info=True)
                raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text strings (batch operation) - SPEED OPTIMIZED.

        Args:
            texts (list): List of input texts to embed.

        Returns:
            list: List of embedding vectors.

        Raises:
            Exception: If embedding fails.

        Edge Cases:
            - Returns cached embeddings where available.
            - Handles empty input list (returns empty list).
        """
        if not texts:
            return []
        
        # SPEED OPTIMIZATION: Truncate texts and check cache
        processed_texts = []
        results = []
        indices_to_compute = []
        
        for i, text in enumerate(texts):
            if not text:
                results.append([0.0] * self.model.get_sentence_embedding_dimension())
                continue
            
            # QUALITY RESTORATION: Balanced text truncation for quality
            if len(text) > 512:
                text = text[:512]
            
            cache_key = f"text:{text}"
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
            else:
                processed_texts.append(text)
                indices_to_compute.append(i)
                results.append(None)  # Placeholder
        
        # Compute only non-cached embeddings
        if processed_texts:
            with self._lock:
                try:
                    # QUALITY RESTORATION: Optimal batch size for quality balance
                    new_embeddings = self.model.encode(
                        processed_texts,
                        show_progress_bar=False,
                        batch_size=64,  # Optimal for quality+speed balance
                        convert_to_tensor=False
                    ).tolist()
                    
                    # Cache and fill results
                    for idx, (text, emb) in enumerate(zip(processed_texts, new_embeddings)):
                        cache_key = f"text:{text}"
                        if len(self._cache) < 5000:
                            self._cache[cache_key] = emb
                        results[indices_to_compute[idx]] = emb
                        
                except Exception as e:
                    self.logger.error(f"Batch embedding failed: {e}", exc_info=True)
                    raise
        
        return results

    # TODO: Add methods for cache management, model reload, and advanced batching if needed.
