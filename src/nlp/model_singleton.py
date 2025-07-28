"""
Model Singleton - Single shared model instance to prevent multiple loadings
"""

import logging
import threading
from typing import Any, Optional, Dict
from sentence_transformers import SentenceTransformer


class ModelSingleton:
    """Singleton class to ensure only one model instance is loaded globally."""
    
    _instance = None
    _lock = threading.Lock()
    _model = None
    _model_name = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> SentenceTransformer:
        """Get the singleton model instance, loading it if necessary."""
        if self._model is None or self._model_name != model_name:
            with self._lock:
                if self._model is None or self._model_name != model_name:
                    logger = logging.getLogger(__name__)
                    logger.info(f"Loading singleton model: {model_name}")
                    
                    self._model = SentenceTransformer(
                        model_name,
                        device="cpu",
                        cache_folder=None,
                        use_auth_token=False,
                        trust_remote_code=False
                    )
                    
                    # Optimize for speed
                    self._model.eval()
                    if hasattr(self._model, 'max_seq_length'):
                        self._model.max_seq_length = 64
                    
                    self._model_name = model_name
                    logger.info("Singleton model loaded successfully")
                    
        return self._model
