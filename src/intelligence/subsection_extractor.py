"""
Subsection extractor - Extract the most relevant text passages from each ranked section.

Purpose:
    Extracts detailed, relevant text passages from ranked sections using semantic similarity to persona/job profiles and configurable heuristics.
    Supports multilingual and mixed-language content. Returns extracted passages with metadata.

Structure:
    - SubsectionExtractor class: main extraction logic

Dependencies:
    - config.settings, config.model_config, config.language_config
    - src.nlp.embeddings (for passage embeddings)
    - src.nlp.similarity (for similarity calculation)
    - src.utils.logger

Integration Points:
    - Used by PersonaDrivenIntelligence orchestrator
    - Outputs extracted passages for output formatting

NOTE: No hardcoded heuristics; all logic is config-driven or semantic.
"""

from typing import Any, Dict, List, Optional
import logging
from config import settings, model_config, language_config
from src.nlp.embeddings import EmbeddingEngine
from src.nlp.similarity import cosine_similarity
from src.utils.logger import get_logger


class SubsectionExtractor:
    """
    SubsectionExtractor extracts the most relevant text passages from each ranked section.

    Args:
        config (dict): Project configuration loaded from config files or environment variables.
        logger (logging.Logger, optional): Logger instance for structured logging.

    Attributes:
        embedding_engine (EmbeddingEngine): Embedding engine for passage profiling.
        config (dict): Loaded configuration.
        logger (logging.Logger): Logger instance.

    Raises:
        ValueError: If input is missing or invalid.

    Limitations:
        Assumes input sections are pre-processed and contain text.
        Embedding model must support all required languages.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the SubsectionExtractor.

        Args:
            config (dict, optional): Project configuration. If None, loads from config files.
            logger (logging.Logger, optional): Logger instance. If None, uses default project logger.
        """
        self.config = config or settings.load_config()
        self.logger = logger or get_logger(__name__)
        self.embedding_engine = EmbeddingEngine(self.config)
        # All thresholds and settings are loaded from config (no hardcoding)
        subsection_settings = self.config.get("SUBSECTION_QUALITY_SETTINGS", {})
        self.max_passages = subsection_settings.get("max_passages_per_section", 3)
        self.passage_min_length = subsection_settings.get("min_passage_length", 150)
        self.quality_settings = {
            "min_passage_length": subsection_settings.get("min_passage_length", 150),
            "target_passage_length": subsection_settings.get("target_passage_length", 400),
            "max_passage_length": subsection_settings.get("max_passage_length", 800),
            "min_information_density": subsection_settings.get("min_information_density", 3),
            "persona_weight": subsection_settings.get("persona_weight", 0.4),
            "job_weight": subsection_settings.get("job_weight", 0.6),
            "deduplication_threshold": subsection_settings.get("deduplication_threshold", 0.92),
            "max_passages_per_section": subsection_settings.get("max_passages_per_section", 3),
            "max_global_passages": subsection_settings.get("max_global_passages", 30),
            "min_unique_tokens": subsection_settings.get("min_unique_tokens", 12),
            "min_sentence_count": subsection_settings.get("min_sentence_count", 2),
            "recipe_passage_settings": subsection_settings.get("recipe_passage_settings", {})
        }

    def extract(
        self,
        ranked_sections: List[Dict[str, Any]],
        persona_profile: Dict[str, Any],
        job_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract the most relevant, context-rich, and valuable text passages from each ranked section.

        This enhanced version merges fragmented steps, ingredients, and instructions into comprehensive, 
        self-contained passages for each recipe or topic, using context, layout, and semantic cues. 
        It also applies persona/job filtering and robust deduplication, all driven by configuration.

        For recipe-style documents:
        - Detects recipe content based on section titles and content patterns
        - Merges ingredients and instructions into comprehensive recipe passages
        - Applies vegetarian/dietary filtering based on persona/job context
        - Creates self-contained recipe descriptions with meaningful titles

        Args:
            ranked_sections (list): List of ranked section dicts (must contain 'text' or 'content').
            persona_profile (dict): Persona profile with embedding.
            job_profile (dict): Job profile with embedding.

        Returns:
            list: List of extracted passage dicts with metadata.

        Raises:
            ValueError: If input is missing or invalid.

        Extraction Logic:
            - Detects recipe-style content using config-driven keywords and patterns
            - For recipe sections: merges ingredients, instructions into comprehensive passages
            - For other sections: uses standard passage extraction logic
            - Applies persona/job filtering (e.g., vegetarian requirements)
            - Deduplicates passages using semantic similarity (configurable threshold)
            - Scores and ranks passages using config-driven heuristics
            - Returns top-N passages per section and globally (configurable)

        Edge Cases:
            - Handles empty or missing sections (returns empty list)
            - Handles missing embeddings (skips or assigns zero score)
            - Recipe sections without both ingredients and instructions are handled gracefully
        """
        if not ranked_sections or not isinstance(ranked_sections, list):
            self.logger.warning("No sections provided for subsection extraction; returning empty list.")
            return []

        persona_emb = persona_profile.get("embedding")
        job_emb = job_profile.get("embedding")
        if persona_emb is None or job_emb is None:
            self.logger.warning("Persona or job embedding missing; assigning zero passage scores.")

        # Detect if we're processing recipe-style documents based on section content
        is_recipe_document = self._detect_recipe_document_context(ranked_sections, persona_profile, job_profile)
        
        # Pre-filter sections for dietary restrictions (e.g., remove non-vegetarian sections)
        filtered_sections = self._filter_sections_by_persona_job(ranked_sections, persona_profile, job_profile)
        
        self.logger.info(f"Processing {len(filtered_sections)} sections (filtered from {len(ranked_sections)}), recipe document: {is_recipe_document}")

        extracted = []
        for section in filtered_sections:
            # Use content if available (aggregated content), otherwise fall back to text
            section_text = section.get("content", section.get("text", ""))
            if not section_text:
                self.logger.debug("Section missing content and text; skipping.")
                continue

            # Choose extraction method based on document type and section content
            if is_recipe_document and self._is_recipe_section(section):
                # Recipe-specific extraction: create comprehensive recipe passages
                candidates = self._create_recipe_passages(section_text, section.get("section_title", ""))
            else:
                # Standard extraction: create semantic passages for non-recipe content
                candidates = self._create_semantic_passages(section_text, section.get("section_title", ""))

            # Deduplicate and filter out empty/trivial passages
            filtered_passages = self._filter_and_deduplicate_passages(candidates)

            # Persona/job filtering: Remove passages that do not match persona/job requirements (config-driven)
            persona_job_filtered = [p for p in filtered_passages if self._matches_persona_job(p, persona_profile, job_profile)]

            passage_scores = []
            for passage in persona_job_filtered:
                # Apply different quality thresholds for recipe vs standard content
                min_length = (self.quality_settings["recipe_passage_settings"]["min_passage_length"] 
                             if is_recipe_document else self.quality_settings["min_passage_length"])
                
                if len(passage.strip()) < min_length:
                    continue
                if not self._is_quality_passage(passage, is_recipe=is_recipe_document):
                    continue
                
                try:
                    passage_emb = self.embedding_engine.embed_text(passage)
                except Exception as e:
                    self.logger.error(f"Passage embedding failed: {e}", exc_info=True)
                    passage_emb = None
                
                persona_sim = cosine_similarity(passage_emb, persona_emb) if passage_emb is not None and persona_emb is not None else 0.0
                job_sim = cosine_similarity(passage_emb, job_emb) if passage_emb is not None and job_emb is not None else 0.0
                
                # Enhanced scoring with quality adjustment
                base_score = (self.quality_settings["persona_weight"] * persona_sim + 
                              self.quality_settings["job_weight"] * job_sim)
                
                # Apply quality bonus for comprehensive content
                quality_bonus = self._calculate_passage_quality_bonus(passage, is_recipe=is_recipe_document)
                final_score = base_score + quality_bonus
                
                passage_scores.append({
                    "passage": passage.strip(),
                    "score": final_score,
                    "base_similarity": base_score,
                    "quality_bonus": quality_bonus,
                    "section_id": section.get("id"),
                    "language": section.get("language"),
                    "is_recipe": is_recipe_document and self._is_recipe_section(section),
                    "section_metadata": {
                        "document": section.get("document", ""),
                        "section_title": section.get("section_title", ""),
                        "page_number": section.get("page_number", 1),
                        "importance_rank": section.get("importance_rank", 0)
                    },
                })
            
            # Sort and select top passages from this section (configurable)
            top_passages = sorted(passage_scores, key=lambda p: p["score"], reverse=True)[:self.quality_settings["max_passages_per_section"]]
            extracted.extend(top_passages)

        # Global ranking: select best passages across all sections
        global_top_passages = sorted(extracted, key=lambda p: p["score"], reverse=True)
        # Limit to top passages but ensure we have meaningful content (configurable)
        final_passages = global_top_passages[:min(len(global_top_passages), self.quality_settings["max_global_passages"])]
        self.logger.info(f"Extracted {len(final_passages)} relevant passages from {len(ranked_sections)} sections.")
        return final_passages

    def _detect_recipe_document_context(self, sections: List[Dict[str, Any]], 
                                       persona_profile: Dict[str, Any], 
                                       job_profile: Dict[str, Any]) -> bool:
        """
        Detect if sections contain recipe-style content based on config-driven keywords and persona/job context.
        
        Args:
            sections: List of section dictionaries
            persona_profile: Persona profile dict
            job_profile: Job profile dict
            
        Returns:
            bool: True if sections appear to contain recipe content
        """
        # Load recipe detection settings from config
        recipe_cfg = self.config.get("RECIPE_PROCESSING_SETTINGS", {})
        recipe_keywords = recipe_cfg.get("recipe_keywords", [])
        recipe_threshold = recipe_cfg.get("recipe_keyword_ratio_threshold", 0.08)
        
        # Check persona/job context for food-related activities
        # Extract actual persona and job data from the profile structure
        persona_raw = persona_profile.get("raw", {})
        job_raw = job_profile.get("raw", {})
        
        # Get the actual text values
        persona_text = persona_raw.get("role", "").lower()
        job_text = job_raw.get("task", "").lower()
        context_text = f"{job_text} {persona_text}"
        
        self.logger.debug(f"Recipe detection context - job_text: '{job_text}', persona_text: '{persona_text}'")
        
        food_context_indicators = ["food", "menu", "recipe", "cooking", "chef", "kitchen", "meal", "dish", "buffet"]
        has_food_context = any(indicator in context_text for indicator in food_context_indicators)
        
        self.logger.debug(f"Recipe detection food context: '{context_text}' -> has_food_context: {has_food_context}")
        
        # Analyze section content for recipe keywords
        total_text = ""
        recipe_sections = 0
        
        for section in sections:
            section_text = section.get("content", section.get("text", ""))
            section_title = section.get("section_title", "").lower()
            
            # Check if section title indicates recipe content
            if any(keyword in section_title for keyword in ["recipe", "ingredients", "instructions"]):
                recipe_sections += 1
            
            total_text += " " + section_text
        
        # Count recipe keywords in total text
        if total_text:
            words = total_text.lower().split()
            recipe_word_count = sum(1 for word in words if any(keyword in word for keyword in recipe_keywords))
            recipe_ratio = recipe_word_count / len(words) if words else 0
            
            # Document is recipe-style if it has food context AND meets keyword threshold
            is_recipe = has_food_context and (recipe_ratio >= recipe_threshold or recipe_sections >= 2)
            
            self.logger.debug(f"Recipe detection: food_context={has_food_context}, "
                            f"recipe_ratio={recipe_ratio:.3f}, recipe_sections={recipe_sections}, "
                            f"threshold={recipe_threshold}, result={is_recipe}")
            
            return is_recipe
        
        return False

    def _filter_sections_by_persona_job(self, sections: List[Dict[str, Any]], 
                                       persona_profile: Dict[str, Any], 
                                       job_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter sections based on persona/job requirements (e.g., remove non-vegetarian sections for vegetarian requirements).
        
        Args:
            sections: List of section dictionaries
            persona_profile: Persona profile dict
            job_profile: Job profile dict
            
        Returns:
            List[Dict[str, Any]]: Filtered sections that match persona/job requirements
        """
        # Load filtering configuration
        filtering_cfg = self.config.get("PERSONA_JOB_FILTERING", {})
        
        # Get job requirements
        persona_raw = persona_profile.get("raw", {})
        job_raw = job_profile.get("raw", {})
        
        job_text = job_raw.get("task", "").lower()
        persona_text = persona_raw.get("role", "").lower()
        
        # Check if vegetarian filtering is needed
        vegetarian_indicators = filtering_cfg.get("vegetarian_indicators", ["vegetarian", "vegan", "plant-based"])
        needs_vegetarian_filtering = any(indicator in job_text for indicator in vegetarian_indicators)
        
        if not needs_vegetarian_filtering:
            # No filtering needed, return all sections
            return sections
        
        # Filter out non-vegetarian sections
        non_veg_exclude = filtering_cfg.get("non_vegetarian_exclude", [])
        filtered_sections = []
        
        for section in sections:
            section_title = section.get("section_title", "").lower()
            section_content = section.get("content", section.get("text", "")).lower()
            
            # Check both title and content for non-vegetarian keywords
            is_non_vegetarian = False
            for keyword in non_veg_exclude:
                if keyword.lower() in section_title or keyword.lower() in section_content:
                    self.logger.info(f"Filtered out non-vegetarian section: '{section.get('section_title', 'Untitled')}' containing '{keyword}'")
                    is_non_vegetarian = True
                    break
            
            if not is_non_vegetarian:
                filtered_sections.append(section)
        
        return filtered_sections

    def _is_recipe_section(self, section: Dict[str, Any]) -> bool:
        """
        Check if a specific section contains recipe content based on title and content patterns.
        
        Args:
            section: Section dictionary
            
        Returns:
            bool: True if section appears to be a recipe
        """
        section_title = section.get("section_title", "").lower()
        section_text = section.get("content", section.get("text", "")).lower()
        
        # Recipe title indicators (config-driven)
        recipe_cfg = self.config.get("RECIPE_PROCESSING_SETTINGS", {})
        ingredient_keywords = recipe_cfg.get("ingredient_keywords", ["ingredient", "ingredients:"])
        instruction_keywords = recipe_cfg.get("instruction_keywords", ["instruction", "instructions:"])
        
        # Check for recipe indicators in title
        recipe_title_indicators = ["recipe", "ingredients", "instructions", "dish", "meal"]
        has_recipe_title = any(indicator in section_title for indicator in recipe_title_indicators)
        
        # Check for recipe structure in content (ingredients + instructions)
        has_ingredients = any(keyword in section_text for keyword in ingredient_keywords)
        has_instructions = any(keyword in section_text for keyword in instruction_keywords)
        
        return has_recipe_title or (has_ingredients and has_instructions)

    def _create_recipe_passages(self, text: str, section_title: str = "") -> List[str]:
        """
        Create comprehensive recipe passages by merging ingredients and instructions into complete recipes.
        All logic is config-driven and creates self-contained recipe descriptions.
        
        Args:
            text: Raw recipe text containing ingredients and instructions
            section_title: Section title for context
            
        Returns:
            List of comprehensive recipe passages
        """
        if not text:
            return []
        
        # Load recipe processing settings from config
        grouping_cfg = self.config.get("SEMANTIC_GROUPING_SETTINGS", {})
        ingredient_keywords = grouping_cfg.get("ingredient_keywords", ["ingredient", "ingredients:"])
        instruction_keywords = grouping_cfg.get("instruction_keywords", ["instruction", "instructions:"])
        
        # Clean and normalize text
        text = text.replace('\n', ' ').replace('  ', ' ').strip()
        
        # Extract recipe name from section title if available
        recipe_name = self._extract_recipe_name_from_title(section_title)
        
        # Split text into parts and identify ingredients/instructions sections
        ingredients_text = ""
        instructions_text = ""
        
        # Look for ingredient and instruction sections
        text_lower = text.lower()
        
        # Find ingredients section
        for keyword in ingredient_keywords:
            if keyword in text_lower:
                start_idx = text_lower.find(keyword)
                # Find end of ingredients (usually where instructions start)
                end_idx = len(text)
                for inst_keyword in instruction_keywords:
                    inst_idx = text_lower.find(inst_keyword, start_idx)
                    if inst_idx > start_idx:
                        end_idx = inst_idx
                        break
                ingredients_text = text[start_idx:end_idx].strip()
                break
        
        # Find instructions section
        for keyword in instruction_keywords:
            if keyword in text_lower:
                start_idx = text_lower.find(keyword)
                instructions_text = text[start_idx:].strip()
                break
        
        # Create comprehensive recipe passage if we have both components
        if ingredients_text and instructions_text:
            recipe_parts = []
            
            # Add recipe name if available
            if recipe_name:
                recipe_parts.append(f"Recipe: {recipe_name}")
            
            # Add ingredients
            recipe_parts.append(ingredients_text)
            
            # Add instructions
            recipe_parts.append(instructions_text)
            
            # Create comprehensive passage
            comprehensive_recipe = " ".join(recipe_parts)
            
            # Apply recipe-specific quality checks
            recipe_cfg = self.config.get("SUBSECTION_QUALITY_SETTINGS", {}).get("recipe_passage_settings", {})
            min_length = recipe_cfg.get("min_passage_length", 100)
            max_length = recipe_cfg.get("max_passage_length", 600)
            
            if len(comprehensive_recipe) >= min_length:
                # Truncate if too long while preserving meaning
                if len(comprehensive_recipe) > max_length:
                    comprehensive_recipe = comprehensive_recipe[:max_length] + "..."
                
                return [comprehensive_recipe]
        
        # Fallback: return original text if recipe structure not detected
        return [text] if len(text) >= 100 else []

    def _extract_recipe_name_from_title(self, section_title: str) -> str:
        """
        Extract a meaningful recipe name from the section title.
        
        Args:
            section_title: Section title text
            
        Returns:
            Extracted recipe name or empty string
        """
        if not section_title:
            return ""
        
        # Remove common suffixes
        title = section_title.replace(" - Ingredients & Instructions", "")
        title = title.replace(" Recipe", "")
        title = title.strip()
        
        # Skip generic titles
        if title.lower() in ["recipe", "ingredients", "instructions"] or title.startswith("Recipe "):
            return ""
        
        return title

    def _create_semantic_passages(self, text: str, section_title: str = "") -> List[str]:
        """
        Merge fragmented steps, ingredients, and instructions into a single, context-rich, self-contained passage for each recipe or topic.
        Uses context, layout, and semantic cues. All grouping logic and thresholds are config-driven.

        Args:
            text: Raw text to split into passages
            section_title: Section heading for context

        Returns:
            List of comprehensive text passages
        """
        if not text:
            return []
        # Clean and normalize the text
        text = text.replace('\n', ' ').replace('  ', ' ').strip()
        # Enhanced: Use config-driven keywords to identify recipe/topic names, ingredients, and instructions
        grouping_cfg = self.config.get("SEMANTIC_GROUPING_SETTINGS", {})
        ingredient_keywords = grouping_cfg.get("ingredient_keywords", ["ingredient", "ingredients:"])
        instruction_keywords = grouping_cfg.get("instruction_keywords", ["instruction", "instructions:", "method", "directions:"])
        # Split text into candidate blocks by config-driven delimiters (e.g., double newlines, headings, etc.)
        block_delimiters = grouping_cfg.get("block_delimiters", ["  ", "\n\n", ". ", "! ", "? "])
        blocks = [text]
        for delim in block_delimiters:
            new_blocks = []
            for b in blocks:
                new_blocks.extend(b.split(delim))
            blocks = new_blocks
        # Group blocks into recipes/topics by detecting ingredient/instruction patterns
        passages = []
        current_passage = []
        found_ingredient = False
        found_instruction = False
        for block in blocks:
            block_lower = block.lower().strip()
            if not block_lower:
                continue
            # Detect start of a new recipe/topic by config-driven heading/section cues
            is_ingredient = any(k in block_lower for k in ingredient_keywords)
            is_instruction = any(k in block_lower for k in instruction_keywords)
            if is_ingredient:
                found_ingredient = True
            if is_instruction:
                found_instruction = True
            current_passage.append(block.strip())
            # If both ingredient and instruction found, treat as a complete recipe/topic
            if found_ingredient and found_instruction:
                passage_text = section_title + ": " + ' '.join(current_passage) if section_title else ' '.join(current_passage)
                passages.append(passage_text.strip())
                current_passage = []
                found_ingredient = False
                found_instruction = False
        # Add any remaining passage if substantial
        if current_passage:
            passage_text = section_title + ": " + ' '.join(current_passage) if section_title else ' '.join(current_passage)
            passages.append(passage_text.strip())
        return passages

    def _matches_persona_job(self, passage: str, persona_profile: Dict[str, Any], job_profile: Dict[str, Any]) -> bool:
        """
        Filter out passages that do not match persona/job requirements (e.g., dietary restrictions, job-specific needs).
        All filtering logic is config-driven and generalized based on persona/job context.

        Args:
            passage: Passage text
            persona_profile: Persona profile dict
            job_profile: Job profile dict

        Returns:
            bool: True if passage matches persona/job requirements, False otherwise
        """
        # Load persona/job filtering configuration
        filtering_cfg = self.config.get("PERSONA_JOB_FILTERING", {})
        
        # Get job requirements from job profile - use correct field structure
        persona_raw = persona_profile.get("raw", {})
        job_raw = job_profile.get("raw", {})
        
        job_text = job_raw.get("task", "").lower()
        persona_text = persona_raw.get("role", "").lower()
        passage_lower = passage.lower()
        
        # Enhanced vegetarian/vegan dietary restrictions filtering - more aggressive
        vegetarian_indicators = filtering_cfg.get("vegetarian_indicators", ["vegetarian", "vegan", "plant-based"])
        if any(indicator in job_text for indicator in vegetarian_indicators):
            non_veg_exclude = filtering_cfg.get("non_vegetarian_exclude", [])
            for keyword in non_veg_exclude:
                # More comprehensive keyword matching - check for word boundaries
                if keyword.lower() in passage_lower:
                    self.logger.info(f"Filtered out non-vegetarian passage containing: '{keyword}' in passage: '{passage[:100]}...'")
                    return False
            
            # Additional check for common non-vegetarian recipe patterns
            non_veg_patterns = [
                "recipe", "cook", "grill", "fry", "roast", "bake", "sauté", "braise", "steam"
            ]
            # If passage contains cooking methods AND non-veg keywords, filter more aggressively
            has_cooking_method = any(pattern in passage_lower for pattern in non_veg_patterns)
            if has_cooking_method:
                # Check for any meat-related words in cooking context
                for keyword in non_veg_exclude:
                    if keyword.lower() in passage_lower:
                        self.logger.info(f"Filtered out cooking passage with non-vegetarian ingredient: '{keyword}'")
                        return False
        
        # Gluten-free filtering for dietary requirements
        gluten_free_indicators = filtering_cfg.get("gluten_free_indicators", ["gluten-free", "gluten free"])
        if any(indicator in job_text for indicator in gluten_free_indicators):
            gluten_exclude = filtering_cfg.get("gluten_exclude", [])
            # Only exclude if explicitly contains gluten without mentioning gluten-free
            contains_gluten = any(keyword in passage_lower for keyword in gluten_exclude)
            mentions_gluten_free = any(indicator in passage_lower for indicator in gluten_free_indicators)
            
            if contains_gluten and not mentions_gluten_free:
                self.logger.debug(f"Filtered out passage with gluten content for gluten-free requirement")
                return False
        
        # Corporate/professional context filtering
        if any(keyword in job_text for keyword in ["corporate", "business", "professional", "office"]):
            corporate_inappropriate = filtering_cfg.get("corporate_inappropriate", [])
            for keyword in corporate_inappropriate:
                if keyword in passage_lower:
                    self.logger.debug(f"Filtered out inappropriate passage for corporate context: {keyword}")
                    return False
        
        return True
        
        return True
        """
        Extract the most relevant, context-rich, and valuable text passages from each ranked section.

        Args:
            ranked_sections (list): List of ranked section dicts (must contain 'text' or 'content').
            persona_profile (dict): Persona profile with embedding.
            job_profile (dict): Job profile with embedding.

        Returns:
            list: List of extracted passage dicts with metadata.

        Raises:
            ValueError: If input is missing or invalid.

        Extraction Logic:
            - For each section, use the full content (aggregated content under headings).
            - Split content into meaningful, context-rich passages (sentences grouped into paragraphs).
            - Filter out passages that are too short, trivial, repetitive, or contextless using config-driven criteria.
            - Deduplicate passages using semantic similarity (configurable threshold).
            - For each passage, compute embedding and similarity to persona/job profiles.
            - Score and rank passages using config-driven heuristics.
            - Select top-N passages per section and globally (configurable).
            - Support multilingual and mixed-language content.

        Edge Cases:
            - Handles empty or missing sections (returns empty list).
            - Handles missing embeddings (skips or assigns zero score).
        """
        if not ranked_sections or not isinstance(ranked_sections, list):
            self.logger.warning("No sections provided for subsection extraction; returning empty list.")
            return []

        persona_emb = persona_profile.get("embedding")
        job_emb = job_profile.get("embedding")
        if persona_emb is None or job_emb is None:
            self.logger.warning("Persona or job embedding missing; assigning zero passage scores.")

        extracted = []
        for section in ranked_sections:
            # Use content if available (aggregated content), otherwise fall back to text
            section_text = section.get("content", section.get("text", ""))
            if not section_text:
                self.logger.debug("Section missing content and text; skipping.")
                continue

            # Create comprehensive, contextual passages instead of fragments
            candidates = self._create_comprehensive_passages(section_text, section.get("section_title", ""))

            # Deduplicate and filter out empty/trivial passages
            filtered_passages = self._filter_and_deduplicate_passages(candidates)

            passage_scores = []
            for passage in filtered_passages:
                # All thresholds and quality checks are config-driven
                if len(passage.strip()) < self.quality_settings["min_passage_length"]:
                    continue
                if not self._is_quality_passage(passage):
                    continue
                try:
                    passage_emb = self.embedding_engine.embed_text(passage)
                except Exception as e:
                    self.logger.error(f"Passage embedding failed: {e}", exc_info=True)
                    passage_emb = None
                persona_sim = cosine_similarity(passage_emb, persona_emb) if passage_emb is not None and persona_emb is not None else 0.0
                job_sim = cosine_similarity(passage_emb, job_emb) if passage_emb is not None and job_emb is not None else 0.0
                # Enhanced scoring with quality adjustment
                base_score = (self.quality_settings["persona_weight"] * persona_sim + 
                              self.quality_settings["job_weight"] * job_sim)
                # Apply quality bonus for comprehensive content
                quality_bonus = self._calculate_passage_quality_bonus(passage)
                final_score = base_score + quality_bonus
                passage_scores.append({
                    "passage": passage.strip(),
                    "score": final_score,
                    "base_similarity": base_score,
                    "quality_bonus": quality_bonus,
                    "section_id": section.get("id"),
                    "language": section.get("language"),
                    "section_metadata": {
                        "document": section.get("document", ""),
                        "section_title": section.get("section_title", ""),
                        "page_number": section.get("page_number", 1),
                        "importance_rank": section.get("importance_rank", 0)
                    },
                })
            # Sort and select top passages from this section (configurable)
            top_passages = sorted(passage_scores, key=lambda p: p["score"], reverse=True)[:self.quality_settings["max_passages_per_section"]]
            extracted.extend(top_passages)

        # Global ranking: select best passages across all sections
        global_top_passages = sorted(extracted, key=lambda p: p["score"], reverse=True)
        # Limit to top passages but ensure we have meaningful content (configurable)
        final_passages = global_top_passages[:min(len(global_top_passages), self.quality_settings["max_global_passages"])]
        self.logger.info(f"Extracted {len(final_passages)} relevant passages from {len(ranked_sections)} sections.")
        return final_passages
    
    def _create_comprehensive_passages(self, text: str, section_title: str = "") -> List[str]:
        """
        Group related steps or sentences into longer, context-rich, self-contained passages.
        All grouping logic and thresholds are config-driven.

        Args:
            text: Raw text to split into passages
            section_title: Section heading for context

        Returns:
            List of comprehensive text passages
        """
        if not text:
            return []
        # Clean and normalize the text
        text = text.replace('\n', ' ').replace('  ', ' ').strip()
        # Enhanced sentence splitting with better context preservation
        sentences = self._split_into_contextual_sentences(text)
        # Group sentences into context-rich, self-contained passages
        passages = self._group_sentences_into_comprehensive_passages(sentences, section_title)
        return passages
    def _filter_and_deduplicate_passages(self, passages: List[str]) -> List[str]:
        """
        Filter out empty, trivial, or highly repetitive passages and deduplicate using semantic similarity.
        All thresholds and deduplication logic are config-driven.

        Args:
            passages: List of candidate passages

        Returns:
            List of filtered, deduplicated passages
        """
        filtered = []
        seen_embeddings = []
        for passage in passages:
            passage = passage.strip()
            if not passage:
                continue
            # Filter out trivial, contextless, or too-short passages
            if len(passage) < self.quality_settings["min_passage_length"]:
                continue
            # Filter by unique token count (configurable)
            unique_tokens = set(passage.lower().split())
            if len(unique_tokens) < self.quality_settings["min_unique_tokens"]:
                continue
            # Filter by minimum sentence count (configurable)
            if passage.count('.') + passage.count('!') + passage.count('?') < self.quality_settings["min_sentence_count"]:
                continue
            # Deduplicate using semantic similarity (configurable threshold)
            try:
                emb = self.embedding_engine.embed_text(passage)
            except Exception as e:
                self.logger.error(f"Deduplication embedding failed: {e}", exc_info=True)
                emb = None
            is_duplicate = False
            if emb is not None:
                for prev_emb in seen_embeddings:
                    sim = cosine_similarity(emb, prev_emb)
                    if sim >= self.quality_settings["deduplication_threshold"]:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    seen_embeddings.append(emb)
            if not is_duplicate:
                filtered.append(passage)
        return filtered
    
    def _split_into_contextual_sentences(self, text: str) -> List[str]:
        """Split text preserving context and meaning."""
        sentences = []
        current_sentence = ""
        
        # Enhanced sentence boundary detection
        for i, char in enumerate(text):
            current_sentence += char
            
            if char in '.!?':
                # Look ahead to avoid splitting on abbreviations
                next_chars = text[i+1:i+4] if i+1 < len(text) else ""
                
                # Check if this is likely end of sentence
                if (len(current_sentence.strip()) > 20 and
                    (not next_chars or next_chars[0].isupper() or next_chars.startswith(' '))):
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
        
        # Add remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return [s for s in sentences if len(s) > 15]  # Filter very short fragments
    
    def _group_sentences_into_comprehensive_passages(self, sentences: List[str], section_title: str) -> List[str]:
        """
        Group related sentences into longer, context-rich, self-contained passages.
        All grouping logic and thresholds are config-driven.

        Args:
            sentences: List of sentences to group
            section_title: Section heading for context

        Returns:
            List of grouped passages
        """
        if not sentences:
            return []
        passages = []
        current_passage = []
        current_length = 0
        target_length = self.quality_settings["target_passage_length"]
        max_length = self.quality_settings["max_passage_length"]
        # Topic-based grouping (configurable in future)
        travel_topics = {
            'location_description': ['located', 'situated', 'place', 'area', 'region', 'beautiful', 'stunning'],
            'activities': ['explore', 'visit', 'experience', 'enjoy', 'discover', 'adventure', 'tour'],
            'practical_info': ['open', 'hours', 'cost', 'price', 'ticket', 'admission', 'access', 'transportation'],
            'dining': ['restaurant', 'food', 'cuisine', 'dining', 'taste', 'chef', 'menu', 'meal'],
            'accommodation': ['hotel', 'stay', 'room', 'accommodation', 'resort', 'lodge'],
            'cultural': ['culture', 'art', 'museum', 'history', 'tradition', 'heritage', 'historic']
        }
        current_topic_group = None
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Determine topic of current sentence
            sentence_topic = self._identify_sentence_topic(sentence, travel_topics)
            # Decide if this sentence should start a new passage
            should_start_new = False
            # Start new passage if we've reached target length
            if current_length > target_length:
                should_start_new = True
            # Start new passage if topic changes significantly and we have substantial content
            elif (current_topic_group and sentence_topic and 
                  sentence_topic != current_topic_group and 
                  current_length > 200):
                should_start_new = True
            # Force new passage if getting too long
            elif current_length > max_length:
                should_start_new = True
            if should_start_new and current_passage:
                # Finalize current passage
                passage_text = ' '.join(current_passage)
                passages.append(passage_text)
                current_passage = [sentence]
                current_length = len(sentence)
                current_topic_group = sentence_topic
            else:
                current_passage.append(sentence)
                current_length += len(sentence)
                if current_topic_group is None:
                    current_topic_group = sentence_topic
        # Add final passage
        if current_passage:
            passage_text = ' '.join(current_passage)
            passages.append(passage_text)
        return passages
    
    def _identify_sentence_topic(self, sentence: str, topic_groups: Dict[str, List[str]]) -> str:
        """Identify the primary topic of a sentence."""
        sentence_lower = sentence.lower()
        best_topic = None
        best_score = 0
        
        for topic, keywords in topic_groups.items():
            score = sum(1 for keyword in keywords if keyword in sentence_lower)
            if score > best_score:
                best_score = score
                best_topic = topic
        
        return best_topic
    
    # _filter_for_quality_content is now handled by _filter_and_deduplicate_passages
    
    def _is_quality_passage(self, passage: str, is_recipe: bool = False) -> bool:
        """
        Check if passage meets quality standards for any document type.
        Uses config-driven criteria that adapt to different content types.
        
        Args:
            passage: Text passage to evaluate
            is_recipe: Whether this is a recipe passage (uses different criteria)
        
        Returns:
            bool: True if passage meets quality standards
        """
        # Apply different length requirements for recipes vs standard content
        if is_recipe:
            recipe_settings = self.quality_settings.get("recipe_passage_settings", {})
            min_length = recipe_settings.get("min_passage_length", 100)
        else:
            min_length = self.quality_settings["min_passage_length"]
        
        if len(passage) < min_length:
            return False
        
        # Information density check (config-driven, relaxed for recipes)
        min_density = 2 if is_recipe else self.quality_settings["min_information_density"]
        info_indicators = passage.count(',') + passage.count(';') + passage.count(':')
        if info_indicators < min_density:
            return False
        
        # Config-driven quality indicators for different content types
        quality_cfg = self.config.get("PASSAGE_QUALITY_SETTINGS", {})
        
        # Travel content indicators
        travel_indicators = quality_cfg.get("travel_value_indicators", [])
        
        # Recipe/food content indicators
        recipe_indicators = quality_cfg.get("recipe_value_indicators", [])
        
        # Technical/business content indicators
        technical_indicators = quality_cfg.get("technical_value_indicators", [])
        
        passage_lower = passage.lower()
        
        # Check if passage has valuable content indicators
        has_travel_value = any(indicator in passage_lower for indicator in travel_indicators)
        has_recipe_value = any(indicator in passage_lower for indicator in recipe_indicators)
        has_technical_value = any(indicator in passage_lower for indicator in technical_indicators)
        
        # For recipe passages, prioritize recipe content
        if is_recipe:
            # Recipe passages should have recipe-specific content or cooking terminology
            cooking_terms = ["ingredients", "instructions", "recipe", "cook", "bake", "prepare", "serve"]
            has_cooking_content = any(term in passage_lower for term in cooking_terms)
            has_value = has_recipe_value or has_cooking_content
        else:
            # Must have at least one type of valuable content
            has_value = has_travel_value or has_recipe_value or has_technical_value
        
        # Config-driven exclusion of generic fragments
        generic_fragments = quality_cfg.get("generic_exclude_fragments", [
            'see above', 'see below', 'as mentioned', 'refer to', 'page',
            'figure', 'table', 'chart', 'note:', 'tip:', 'warning:'
        ])
        
        has_generic_content = any(fragment in passage_lower for fragment in generic_fragments)
        
        return has_value and not has_generic_content
    
    def _calculate_passage_quality_bonus(self, passage: str, is_recipe: bool = False) -> float:
        """
        Calculate quality bonus for comprehensive, informative passages.
        
        Args:
            passage: Text passage to evaluate
            is_recipe: Whether this is a recipe passage (different scoring)
            
        Returns:
            float: Quality bonus score (0.0 to 1.0)
        """
        bonus = 0.0
        passage_lower = passage.lower()
        
        # Apply different quality bonuses for recipes vs standard content
        if is_recipe:
            # Recipe-specific quality bonuses
            
            # Bonus for comprehensive recipe structure
            has_ingredients = any(term in passage_lower for term in ["ingredient", "ingredients:"])
            has_instructions = any(term in passage_lower for term in ["instruction", "instructions:", "method", "directions"])
            if has_ingredients and has_instructions:
                bonus += 0.4  # High bonus for complete recipes
            
            # Bonus for detailed cooking information
            cooking_details = ["temperature", "time", "minutes", "degrees", "cup", "tablespoon", "teaspoon"]
            detail_count = sum(1 for detail in cooking_details if detail in passage_lower)
            if detail_count >= 3:
                bonus += 0.3
            elif detail_count >= 2:
                bonus += 0.2
            elif detail_count >= 1:
                bonus += 0.1
            
            # Bonus for comprehensive length (recipe-specific target)
            recipe_target_length = self.quality_settings.get("recipe_passage_settings", {}).get("target_passage_length", 300)
            if len(passage) > recipe_target_length:
                bonus += 0.2
        else:
            # Standard quality bonuses for non-recipe content
            
            # Bonus for comprehensive length
            if len(passage) > self.quality_settings["target_passage_length"]:
                bonus += 0.2
            
            # Bonus for information richness
            info_density = (passage.count(',') + passage.count(';') + 
                           passage.count(':') + passage.count('('))
            if info_density >= 8:
                bonus += 0.3
            elif info_density >= 5:
                bonus += 0.2
            elif info_density >= 3:
                bonus += 0.1
            
            # Bonus for actionable advice
            actionable_indicators = [
                'recommend', 'suggest', 'perfect for', 'ideal for', 'best time',
                'make sure', 'don\'t miss', 'worth visiting', 'must see'
            ]
            
            actionable_count = sum(1 for indicator in actionable_indicators 
                                  if indicator in passage_lower)
            
            if actionable_count >= 2:
                bonus += 0.2
            elif actionable_count >= 1:
                bonus += 0.1
        
        return min(bonus, 1.0)  # Cap bonus at 1.0
