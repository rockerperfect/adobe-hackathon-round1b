"""
Persona analyzer - Parse and profile personas for persona-driven document intelligence.

Purpose:
    Parses persona information from input JSON and generates a semantic persona profile (embedding or feature vector) for downstream ranking.

Structure:
    - PersonaAnalyzer class: main persona analysis logic

Dependencies:
    - config.settings, config.model_config, config.language_config
    - src.nlp.embeddings (for embedding generation)
    - src.utils.logger

Integration Points:
    - Used by PersonaDrivenIntelligence orchestrator
    - Outputs persona profile for section ranking and relevance calculation

NOTE: No hardcoded persona types; all logic is config-driven or semantic.
"""

from typing import Any, Dict, Optional
import logging
from config import settings, model_config, language_config
from src.nlp.embeddings import EmbeddingEngine
from src.utils.logger import get_logger


class PersonaAnalyzer:
    """
    PersonaAnalyzer parses persona information and generates a semantic persona profile.

    Args:
        config (dict): Project configuration loaded from config files or environment variables.
        logger (logging.Logger, optional): Logger instance for structured logging.

    Attributes:
        embedding_engine (EmbeddingEngine): Embedding engine for semantic profiling.
        config (dict): Loaded configuration.
        logger (logging.Logger): Logger instance.

    Raises:
        ValueError: If persona information is missing or invalid.

    Limitations:
        Assumes input JSON is pre-validated.
        Embedding model must support all required languages.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the PersonaAnalyzer.

        Args:
            config (dict, optional): Project configuration. If None, loads from config files.
            logger (logging.Logger, optional): Logger instance. If None, uses default project logger.
        """
        self.config = config or settings.load_config()
        self.logger = logger or get_logger(__name__)
        self.embedding_engine = EmbeddingEngine(self.config)

    def analyze(self, persona_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse persona information and generate a semantic persona profile.

        Args:
            persona_context (dict): Persona information from input JSON.

        Returns:
            dict: Persona profile including embedding/feature vector and metadata.

        Raises:
            ValueError: If persona information is missing or invalid.

        Edge Cases:
            - Handles multilingual persona descriptions.
            - Returns default/empty profile if persona context is empty (with warning).
        """
        if not persona_context or not isinstance(persona_context, dict):
            self.logger.warning("Persona context missing or invalid; returning default profile.")
            # TODO: Consider raising or handling as per config
            return {"embedding": None, "raw": {}, "metadata": {"warning": "No persona context provided."}}

        # Extract persona description (supporting multilingual/cross-domain)
        description = persona_context.get("description")
        if not description:
            self.logger.warning("Persona description missing; using full context as description.")
            description = str(persona_context)

        # Generate embedding/feature vector
        try:
            embedding = self.embedding_engine.embed_text(description)
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}", exc_info=True)
            embedding = None

        profile = {
            "embedding": embedding,
            "raw": persona_context,
            "metadata": {
                "language": persona_context.get("language"),
                "source": "PersonaAnalyzer",
            },
        }
        self.logger.info("Persona profile generated.")
        return profile
