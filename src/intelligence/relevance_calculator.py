"""
Relevance calculator - Calculate detailed importance/relevance scores for sections and subsections.

Purpose:
    Calculates normalized relevance scores for sections and subsections using semantic similarity, persona/job weighting, and additional config-driven signals.

Structure:
    - RelevanceCalculator class: main scoring logic

Dependencies:
    - config.settings, config.model_config
    - src.nlp.similarity (for similarity calculation)
    - src.utils.logger

Integration Points:
    - Used by PersonaDrivenIntelligence orchestrator
    - Outputs normalized scores for downstream modules

NOTE: No hardcoded weights; all weighting is config-driven.
"""

from typing import Any, Dict, List, Optional
import logging
from config import settings, model_config
from src.nlp.similarity import cosine_similarity
from src.utils.logger import get_logger


class RelevanceCalculator:
    """
    RelevanceCalculator computes normalized importance/relevance scores for sections and subsections.

    Args:
        config (dict): Project configuration loaded from config files or environment variables.
        logger (logging.Logger, optional): Logger instance for structured logging.

    Attributes:
        config (dict): Loaded configuration.
        logger (logging.Logger): Logger instance.

    Raises:
        ValueError: If input is missing or invalid.

    Limitations:
        Assumes input sections/subsections are pre-processed and contain embeddings.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the RelevanceCalculator.

        Args:
            config (dict, optional): Project configuration. If None, loads from config files.
            logger (logging.Logger, optional): Logger instance. If None, uses default project logger.
        """
        self.config = config or settings.load_config()
        self.logger = logger or get_logger(__name__)
        self.weights = self.config.get("relevance_weights", {"persona": 0.5, "job": 0.5, "other": 0.0})

    def calculate(
        self,
        ranked_sections: List[Dict[str, Any]],
        persona_profile: Dict[str, Any],
        job_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Calculate normalized relevance scores for sections and subsections.

        Args:
            ranked_sections (list): List of ranked section dicts (must contain 'embedding').
            persona_profile (dict): Persona profile with embedding.
            job_profile (dict): Job profile with embedding.

        Returns:
            list: List of section dicts with added 'relevance_score' field (normalized 0-1).

        Raises:
            ValueError: If input is missing or invalid.

        Scoring Logic:
            - For each section, compute similarity to persona and job embeddings.
            - Combine using adaptive weights from config.
            - Optionally include other signals (e.g., section length, metadata).
            - Normalize all scores to 0-1 range.

        Edge Cases:
            - Handles empty or missing sections (returns empty list).
            - Handles missing embeddings (assigns zero score).
        """
        if not ranked_sections or not isinstance(ranked_sections, list):
            self.logger.warning("No sections provided for relevance calculation; returning empty list.")
            return []

        persona_emb = persona_profile.get("embedding")
        job_emb = job_profile.get("embedding")
        if persona_emb is None or job_emb is None:
            self.logger.warning("Persona or job embedding missing; assigning zero relevance scores.")

        scores = []
        for section in ranked_sections:
            section_emb = section.get("embedding")
            if section_emb is None:
                self.logger.debug("Section missing embedding; assigning zero relevance score.")
                score = 0.0
            else:
                persona_sim = cosine_similarity(section_emb, persona_emb) if persona_emb is not None else 0.0
                job_sim = cosine_similarity(section_emb, job_emb) if job_emb is not None else 0.0
                # Optionally add other signals (e.g., section length)
                other_signal = section.get("other_signal", 0.0)
                score = (
                    self.weights["persona"] * persona_sim +
                    self.weights["job"] * job_sim +
                    self.weights.get("other", 0.0) * other_signal
                )
            scores.append(score)

        # Normalize scores to 0-1
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 1.0
        def normalize(val: float) -> float:
            if max_score == min_score:
                return 1.0 if val > 0 else 0.0
            return (val - min_score) / (max_score - min_score)

        for i, section in enumerate(ranked_sections):
            section["relevance_score"] = normalize(scores[i])

        self.logger.info(f"Calculated normalized relevance scores for {len(ranked_sections)} sections.")
        return ranked_sections
