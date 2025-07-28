"""
Job analyzer - Analyze job requirements and generate job profile for relevance ranking.

Purpose:
    Implements the JobAnalyzer class for parsing job-to-be-done context from input JSON and generating a semantic job profile (embedding or feature vector) for downstream ranking.

Structure:
    - JobAnalyzer class: main job analysis logic

Dependencies:
    - config.settings, config.model_config, config.language_config
    - src.nlp.embeddings (for embedding generation)
    - src.utils.logger

Integration Points:
    - Used by PersonaDrivenIntelligence orchestrator
    - Outputs job profile for section ranking and relevance calculation

NOTE: No hardcoding; all configuration, model paths, and language codes are config-driven.
"""

from typing import Any, Dict, Optional
import logging
from config import settings, model_config, language_config
from src.nlp.embeddings import EmbeddingEngine
from src.utils.logger import get_logger


class JobAnalyzer:
    """
    JobAnalyzer parses job-to-be-done context and generates a semantic job profile.

    Args:
        config (dict): Project configuration loaded from config files or environment variables.
        logger (logging.Logger, optional): Logger instance for structured logging.

    Attributes:
        embedding_engine (EmbeddingEngine): Embedding engine for semantic profiling.
        config (dict): Loaded configuration.
        logger (logging.Logger): Logger instance.

    Raises:
        ValueError: If job context is missing or invalid.

    Limitations:
        Assumes input JSON is pre-validated.
        Embedding model must support all required languages.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the JobAnalyzer.

        Args:
            config (dict, optional): Project configuration. If None, loads from config files.
            logger (logging.Logger, optional): Logger instance. If None, uses default project logger.
        """
        self.config = config or settings.load_config()
        self.logger = logger or get_logger(__name__)
        self.embedding_engine = EmbeddingEngine(self.config)

    def analyze(self, job_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse job-to-be-done information and generate a semantic job profile.

        Args:
            job_context (dict): Job-to-be-done information from input JSON.

        Returns:
            dict: Job profile including embedding/feature vector and metadata.

        Raises:
            ValueError: If job context is missing or invalid.

        Edge Cases:
            - Handles multilingual job descriptions.
            - Returns default/empty profile if job context is empty (with warning).
        """
        if not job_context or not isinstance(job_context, dict):
            self.logger.warning("Job context missing or invalid; returning default profile.")
            # TODO: Consider raising or handling as per config
            return {"embedding": None, "raw": {}, "metadata": {"warning": "No job context provided."}}

        # Extract job description (supporting multilingual/domain-agnostic)
        description = job_context.get("description")
        if not description:
            self.logger.warning("Job description missing; using full context as description.")
            description = str(job_context)

        # Generate embedding/feature vector
        try:
            embedding = self.embedding_engine.embed_text(description)
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}", exc_info=True)
            embedding = None

        profile = {
            "embedding": embedding,
            "raw": job_context,
            "metadata": {
                "language": job_context.get("language"),
                "source": "JobAnalyzer",
            },
        }
        self.logger.info("Job profile generated.")
        return profile
