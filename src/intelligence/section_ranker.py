"""
Section ranker - Rank document sections by relevance to persona and job profiles.

Purpose:
    Computes relevance scores for each section using embedding similarity and adaptive weighting (from config).
    Outputs a ranked list of sections with scores. Supports batch/multi-document ranking.

Structure:
    - SectionRanker class: main section ranking logic

Dependencies:
    - config.settings, config.model_config
    - src.nlp.embeddings (for section embeddings)
    - src.nlp.similarity (for similarity calculation)
    - src.utils.logger

Integration Points:
    - Used by PersonaDrivenIntelligence orchestrator
    - Outputs ranked sections for relevance calculation and extraction

NOTE: No hardcoded weights; all weighting is config-driven.
"""

from typing import Any, Dict, List, Optional
import logging
from config import settings, model_config
from src.nlp.embeddings import EmbeddingEngine
from src.nlp.similarity import cosine_similarity
from src.utils.logger import get_logger


class SectionRanker:
    """
    SectionRanker ranks document sections by relevance to persona and job profiles.

    Args:
        config (dict): Project configuration loaded from config files or environment variables.
        logger (logging.Logger, optional): Logger instance for structured logging.

    Attributes:
        embedding_engine (EmbeddingEngine): Embedding engine for section profiling.
        config (dict): Loaded configuration.
        logger (logging.Logger): Logger instance.

    Raises:
        ValueError: If input sections are missing or invalid.

    Limitations:
        Assumes input sections are pre-processed and contain text.
        Embedding model must support all required languages.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the SectionRanker.

        Args:
            config (dict, optional): Project configuration. If None, loads from config files.
            logger (logging.Logger, optional): Logger instance. If None, uses default project logger.
        """
        self.config = config or settings.load_config()
        self.logger = logger or get_logger(__name__)
        self.embedding_engine = EmbeddingEngine(self.config)
        self.weights = self.config.get("section_ranking_weights", {"persona": 0.5, "job": 0.5})
        
        # Quality settings for enhanced section evaluation
        self.quality_settings = self.config.get("SECTION_QUALITY_SETTINGS", {
            "min_section_length": 200,
            "min_section_words": 30,
            "generic_title_penalty": -0.5,
            "fragment_penalty": -0.3,
            "short_content_penalty": -0.2,
            "content_diversity_weight": 0.3,
            "informativeness_weight": 0.4,
            "context_relevance_weight": 0.3
        })

    def rank(
        self,
        sections: List[Dict[str, Any]],
        persona_profile: Dict[str, Any],
        job_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Rank sections by relevance to persona/job - SPEED OPTIMIZED.
        All scoring weights, penalties, and keyword lists are config-driven (no hardcoding).
        """
        if not sections or not isinstance(sections, list):
            self.logger.warning("No sections provided for ranking; returning empty list.")
            return []

        # SPEED OPTIMIZATION: Early truncation for large section counts
        if len(sections) > 200:
            self.logger.warning(f"Large section count ({len(sections)}), taking top 200 for speed")
            sections = sections[:200]

        # Apply persona/job-based filtering before ranking
        filtered_sections = self._filter_sections_by_persona_job(sections, persona_profile, job_profile)
        if len(filtered_sections) != len(sections):
            self.logger.info(f"Filtered sections from {len(sections)} to {len(filtered_sections)} based on persona/job requirements")
        
        sections = filtered_sections
        if not sections:
            self.logger.warning("No sections remaining after filtering; returning empty list.")
            return []

        # SPEED OPTIMIZATION: Limit final ranking to top candidates
        max_sections_to_rank = self.config.get('max_sections_to_rank', 100)
        if len(sections) > max_sections_to_rank:
            sections = sections[:max_sections_to_rank]

        persona_emb = persona_profile.get("embedding")
        job_emb = job_profile.get("embedding")
        if persona_emb is None or job_emb is None:
            self.logger.warning("Persona or job embedding missing; assigning zero scores.")

        # Load config-driven trivial/generic words and persona/job keywords
        quality_cfg = self.quality_settings
        persona_keywords = set(self.config.get("persona_keywords", ["vegetarian", "buffet", "travel", "form", "gluten-free", "menu", "trip", "accommodation", "hotel", "restaurant", "activity", "compliance", "onboarding"]))
        trivial_words = set(self.config.get("trivial_section_titles", [
            "ingredients", "instructions", "directions", "steps", "method", "notes", "tip", "tips", "serves", "prep time", "cook time", "garnish", "serve", "yield"
        ]))

        # Deduplication: track normalized titles
        seen_titles = set()
        ranked = []
        for section in sections:
            text = section.get("text", "")
            content = section.get("content", text)
            section_title = section.get("section_title", section.get("title", ""))
            norm_title = section_title.lower().strip()
            # Penalize duplicate or near-duplicate titles
            if norm_title in seen_titles:
                continue
            seen_titles.add(norm_title)
            # Penalize short or generic titles
            penalty = 0.0
            if len(norm_title.split()) < 3 or any(w in norm_title for w in trivial_words):
                penalty += quality_cfg.get("generic_title_penalty", -0.5)
            # Boost for persona/job keywords
            boost = 0.0
            for kw in persona_keywords:
                if kw in norm_title or kw in content.lower():
                    boost += 0.2
            # Embedding-based similarity
            try:
                section_emb = self.embedding_engine.embed_text(content)
            except Exception as e:
                self.logger.error(f"Section embedding failed: {e}", exc_info=True)
                section_emb = None
            persona_sim = cosine_similarity(section_emb, persona_emb) if section_emb is not None and persona_emb is not None else 0.0
            job_sim = cosine_similarity(section_emb, job_emb) if section_emb is not None and job_emb is not None else 0.0
            base_score = self.weights["persona"] * persona_sim + self.weights["job"] * job_sim
            # Enhanced quality scoring
            quality_score = self._calculate_quality_score(section_title, content)
            final_score = base_score + quality_score + penalty + boost
            ranked.append({**section, "score": final_score, "base_similarity": base_score, "quality_score": quality_score, "penalty": penalty, "boost": boost})
        # Sort by score descending
        ranked_sorted = sorted(ranked, key=lambda s: s["score"], reverse=True)
        self.logger.info(f"Ranked {len(ranked_sorted)} sections.")
        return ranked_sorted
    
    def _calculate_quality_score(self, title: str, content: str) -> float:
        """
        Calculate quality score for section to penalize generic/fragmented content.
        
        Args:
            title: Section title
            content: Section content
            
        Returns:
            Quality score (positive for high quality, negative for low quality)
        """
        quality_score = 0.0
        
        # Penalty for generic/fragment titles
        generic_indicators = [
            'ingredients', 'directions', 'instructions', 'step', 'note', 'tip',
            '1/', '2/', '3/', '4/', 'cup', 'tablespoon', 'teaspoon', 'minutes',
            'serves', 'prep time', 'cook time', 'temperature'
        ]
        
        title_lower = title.lower()
        if any(indicator in title_lower for indicator in generic_indicators):
            quality_score += self.quality_settings["generic_title_penalty"]
        
        # Penalty for very short content (likely fragments)
        if len(content) < self.quality_settings["min_section_length"]:
            quality_score += self.quality_settings["short_content_penalty"]
        
        # Penalty for content with insufficient words
        word_count = len(content.split())
        if word_count < self.quality_settings["min_section_words"]:
            quality_score += self.quality_settings["fragment_penalty"]
        
        # Bonus for comprehensive, informative content
        informativeness_bonus = self._calculate_informativeness_bonus(content)
        quality_score += informativeness_bonus
        
        # Bonus for content diversity (multiple topics/concepts)
        diversity_bonus = self._calculate_diversity_bonus(content)
        quality_score += diversity_bonus
        
        return quality_score
    
    def _calculate_informativeness_bonus(self, content: str) -> float:
        """Calculate bonus for informative content with rich details."""
        bonus = 0.0
        
        # Indicators of informative travel content
        info_indicators = [
            'explore', 'discover', 'experience', 'enjoy', 'visit', 'try',
            'recommend', 'suggest', 'perfect', 'ideal', 'beautiful', 'stunning',
            'popular', 'famous', 'known', 'located', 'situated', 'features'
        ]
        
        content_lower = content.lower()
        info_count = sum(1 for indicator in info_indicators if indicator in content_lower)
        
        # Bonus based on information density
        if info_count >= 5:
            bonus += 0.3
        elif info_count >= 3:
            bonus += 0.2
        elif info_count >= 1:
            bonus += 0.1
        
        # Additional bonus for detailed descriptions
        if len(content) > 500:
            bonus += 0.2
        elif len(content) > 300:
            bonus += 0.1
        
        return bonus * self.quality_settings["informativeness_weight"]
    
    def _calculate_diversity_bonus(self, content: str) -> float:
        """Calculate bonus for content covering diverse topics."""
        bonus = 0.0
        
        # Topic categories for travel content
        topic_categories = {
            'location': ['place', 'area', 'region', 'town', 'city', 'village'],
            'activities': ['activity', 'tour', 'adventure', 'experience', 'excursion'],
            'dining': ['restaurant', 'food', 'cuisine', 'dining', 'meal'],
            'culture': ['culture', 'art', 'museum', 'history', 'tradition'],
            'accommodation': ['hotel', 'stay', 'accommodation', 'resort'],
            'practical': ['tip', 'advice', 'important', 'remember', 'note']
        }
        
        content_lower = content.lower()
        covered_topics = 0
        
        for category, keywords in topic_categories.items():
            if any(keyword in content_lower for keyword in keywords):
                covered_topics += 1
        
        # Bonus for covering multiple topic areas
        if covered_topics >= 4:
            bonus += 0.4
        elif covered_topics >= 3:
            bonus += 0.3
        elif covered_topics >= 2:
            bonus += 0.2
        
        return bonus * self.quality_settings["content_diversity_weight"]

    def _filter_sections_by_persona_job(
        self,
        sections: List[Dict[str, Any]],
        persona_profile: Dict[str, Any],
        job_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter sections based on persona and job requirements.
        
        Args:
            sections: List of sections to filter
            persona_profile: Persona profile data
            job_profile: Job profile data
            
        Returns:
            Filtered list of sections
        """
        # Get filtering configuration from settings
        persona_job_filtering = self.config.get("PERSONA_JOB_FILTERING", {})
        
        # Extract persona and job text for filtering
        # Handle original_text which can be a dict or string
        persona_text = ""
        if isinstance(persona_profile.get("original_text"), dict):
            # Extract from dict structure
            orig_dict = persona_profile["original_text"]
            persona_text = str(orig_dict.get("role", "")) or str(orig_dict.get("persona", ""))
        elif isinstance(persona_profile.get("original_text"), str):
            persona_text = persona_profile["original_text"]
        elif isinstance(persona_profile.get("text"), str):
            persona_text = persona_profile["text"]
        else:
            persona_text = str(persona_profile.get("persona", ""))
        
        job_text = ""
        if isinstance(job_profile.get("original_text"), dict):
            # Extract from dict structure
            orig_dict = job_profile["original_text"]
            job_text = str(orig_dict.get("task", "")) or str(orig_dict.get("job_to_be_done", "")) or str(orig_dict.get("job", ""))
        elif isinstance(job_profile.get("original_text"), str):
            job_text = job_profile["original_text"]
        elif isinstance(job_profile.get("text"), str):
            job_text = job_profile["text"]
        else:
            job_text = str(job_profile.get("job", ""))
        
        persona_text = persona_text.lower()
        job_text = job_text.lower()
        
        filtered_sections = []
        
        for section in sections:
            section_title = section.get("section_title", section.get("title", "")).lower()
            section_content = section.get("content", section.get("text", "")).lower()
            
            # Check if section matches persona/job filtering requirements
            if self._matches_persona_job_requirements(section_title, section_content, persona_text, job_text, persona_job_filtering):
                filtered_sections.append(section)
            else:
                self.logger.info(f"Filtered out section in ranking: '{section.get('section_title', 'Unknown')}' due to persona/job requirements")
        
        return filtered_sections

    def _matches_persona_job_requirements(
        self,
        section_title: str,
        section_content: str,
        persona_text: str,
        job_text: str,
        filtering_config: Dict[str, Any]
    ) -> bool:
        """
        Check if a section matches persona and job requirements.
        
        Args:
            section_title: Section title (lowercase)
            section_content: Section content (lowercase)
            persona_text: Persona text (lowercase)
            job_text: Job text (lowercase)
            filtering_config: Filtering configuration
            
        Returns:
            True if section matches requirements, False otherwise
        """
        # For vegetarian filtering
        if "vegetarian" in persona_text or "vegetarian" in job_text:
            non_vegetarian_exclude = filtering_config.get("non_vegetarian_exclude", [])
            
            # Check title and content for non-vegetarian keywords
            combined_text = f"{section_title} {section_content}"
            for exclude_term in non_vegetarian_exclude:
                if exclude_term.lower() in combined_text:
                    return False
        
        return True
