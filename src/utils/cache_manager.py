"""
Cache manager - NEW: Model and embedding caching for Round 1B.

Purpose:
    Implements the CacheManager class for caching models and embeddings with configurable size, eviction policy, and persistence options.

Structure:
    - CacheManager class: main caching logic

Dependencies:
    - config.settings (for cache config)
    - src.utils.logger (for logging)
    - threading, collections

Integration Points:
    - Used by model and embedding engines for caching
    - Supports cache size, eviction, and persistence

NOTE: No hardcoding; all cache config is loaded dynamically.
"""


from typing import Any, Dict, Optional
import threading
import logging
from collections import OrderedDict
import os
from config import settings
from src.utils.logger import get_logger


class CacheManager:
    """
    CacheManager provides model and embedding caching with configurable size and eviction policy.

    Args:
        config (dict): Project configuration loaded from config files or environment variables.
        logger (logging.Logger, optional): Logger instance for structured logging.

    Attributes:
        config (dict): Loaded configuration.
        logger (logging.Logger): Logger instance.
        _cache (OrderedDict): Internal cache storage.
        _lock (threading.Lock): Thread lock for safe access.
        max_size (int): Maximum cache size.
        persistence_path (str, optional): Path for cache persistence.

    Raises:
        ValueError: If cache config is invalid.

    Limitations:
        Persistence is optional and not enabled by default.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the CacheManager.

        Args:
            config (dict, optional): Project configuration. If None, loads from settings.
            logger (logging.Logger, optional): Logger instance. If None, uses default project logger.
        """
        self.config = config or settings.load_config()
        self.logger = logger or get_logger(__name__)
        self.max_size = self.config.get('cache_max_size', 1000)
        self.persistence_path = self.config.get('cache_persistence_path')
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def cache(self, key: str, value: Any) -> None:
        """
        Cache a value by key, evicting oldest if over max_size.

        Args:
            key (str): Cache key.
            value (Any): Value to cache.

        Raises:
            ValueError: If key is missing.
        """
        if not key:
            raise ValueError("Cache key is required.")
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self.max_size:
                evicted = self._cache.popitem(last=False)
                self.logger.info(f"Cache evicted: {evicted[0]}")

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached value by key.

        Args:
            key (str): Cache key.

        Returns:
            Any or None: Cached value or None if not found.
        """
        with self._lock:
            value = self._cache.get(key)
            if value is not None:
                self._cache.move_to_end(key)
            return value

    def invalidate(self, key: str) -> None:
        """
        Invalidate a cached value by key.

        Args:
            key (str): Cache key.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self.logger.info(f"Cache invalidated: {key}")

    def clear(self) -> None:
        """
        Clear the entire cache.

        Returns:
            None
        """
        with self._lock:
            self._cache.clear()
            self.logger.info("Cache cleared.")

    def persist(self) -> None:
        """
        Persist the cache to disk if persistence_path is set.

        Returns:
            None
        """
        if not self.persistence_path:
            self.logger.warning("Cache persistence path not set; skipping persistence.")
            return
        try:
            import pickle
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(self._cache, f)
            self.logger.info(f"Cache persisted to {self.persistence_path}")
        except Exception as e:
            self.logger.error(f"Failed to persist cache: {e}", exc_info=True)

    def load(self) -> None:
        """
        Load the cache from disk if persistence_path is set.

        Returns:
            None
        """
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            self.logger.warning("Cache persistence file not found; skipping load.")
            return
        try:
            import pickle
            with open(self.persistence_path, 'rb') as f:
                self._cache = pickle.load(f)
            self.logger.info(f"Cache loaded from {self.persistence_path}")
        except Exception as e:
            self.logger.error(f"Failed to load cache: {e}", exc_info=True)

    # TODO: Add support for advanced eviction policies and distributed caching if needed.
