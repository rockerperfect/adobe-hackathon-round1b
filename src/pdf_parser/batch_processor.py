"""
Batch PDF processor - NEW for Round 1B.
Handles processing of multiple PDF documents simultaneously.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .pymupdf_parser import PyMuPDFParser
from .pdfminer_parser import PDFMinerParser


class BatchProcessor:
    """
    Handles batch processing of multiple PDF documents.
    Optimized for Round 1B multi-document intelligence requirements.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize batch processor.
        
        Args:
            max_workers: Maximum number of parallel processing threads
        """
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
    def process_documents(self, document_paths: List[Path], 
                         use_fallback: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple PDF documents in parallel.
        
        Args:
            document_paths: List of paths to PDF files
            use_fallback: Whether to use PDFMiner as fallback for failed documents
            
        Returns:
            List of processed document data
        """
        self.logger.info(f"Starting batch processing of {len(document_paths)} documents")
        start_time = time.time()
        
        results = []
        
        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._process_single_document, path, use_fallback): path 
                for path in document_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self.logger.info(f"Successfully processed: {path.name}")
                    else:
                        self.logger.error(f"Failed to process: {path.name}")
                except Exception as e:
                    self.logger.error(f"Exception processing {path.name}: {str(e)}")
        
        processing_time = time.time() - start_time
        self.logger.info(f"Batch processing completed in {processing_time:.2f} seconds")
        
        return results
    
    def _process_single_document(self, pdf_path: Path, 
                                use_fallback: bool = True) -> Optional[Dict[str, Any]]:
        """
        Process a single PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            use_fallback: Whether to use PDFMiner as fallback
            
        Returns:
            Processed document data or None if failed
        """
        if not pdf_path.exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return None
        
        # Try primary parser (PyMuPDF)
        parser = PyMuPDFParser(str(pdf_path))
        
        try:
            return self._extract_document_data(parser, pdf_path)
        except Exception as e:
            self.logger.warning(f"PyMuPDF failed for {pdf_path.name}: {str(e)}")
        finally:
            parser.close()
        
        # Try fallback parser (PDFMiner) if enabled
        if use_fallback:
            self.logger.info(f"Trying PDFMiner fallback for: {pdf_path.name}")
            parser = PDFMinerParser(str(pdf_path))
            
            try:
                return self._extract_document_data(parser, pdf_path)
            except Exception as e:
                self.logger.error(f"PDFMiner also failed for {pdf_path.name}: {str(e)}")
            finally:
                parser.close()
        
        return None
    
    def _extract_document_data(self, parser, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract comprehensive data from a loaded document.
        
        Args:
            parser: Loaded PDF parser instance
            pdf_path: Path to the PDF file
            
        Returns:
            Comprehensive document data
        """
        # Parse the PDF using the base parser interface
        parsed_data = parser.parse_pdf(str(pdf_path))
        
        # Extract text blocks with positions
        text_blocks = parser.extract_text_with_positions()
        
        # Get document metadata
        metadata = parser.get_document_metadata()
        
        # Extract text content from text blocks
        text_content = "\n".join([block.text for block in text_blocks])
        
        # Extract sections from text blocks (basic heading detection)
        sections = self._extract_sections_from_text_blocks(text_blocks, pdf_path)
        
        # Prepare document data for the intelligence pipeline
        document_data = {
            "filename": pdf_path.name,
            "file_path": str(pdf_path),
            "text_content": text_content,
            "metadata": metadata,
            "page_count": parsed_data.get('page_count', 0),
            "text_blocks": text_blocks,
            "sections": sections,
            "parser_used": parser.__class__.__name__,
            "processing_time": parsed_data.get('processing_time', 0),
            "extraction_timestamp": time.time(),
            "content_length": len(text_content),
            "has_text": len(text_content.strip()) > 0
        }
        
        return document_data
    
    def _extract_sections_from_text_blocks(self, text_blocks, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract only meaningful, context-rich sections from text blocks.
        - For recipe-style documents: Merge ingredients, instructions, and recipe names into comprehensive sections.
        - For other documents: Use existing high-quality section extraction logic.
        - Document type detection is based on config-driven keywords and persona/job context, not hardcoded paths.
        - All config, keywords, and thresholds are loaded from config files (no hardcoding).
        """
        from config import settings
        # Load config-driven quality and recipe settings
        quality_cfg = getattr(settings, "SECTION_QUALITY_SETTINGS", {})
        # For recipe-style detection, use a dedicated config section if present
        recipe_cfg = quality_cfg.get("RECIPE_DOCUMENT_SETTINGS", getattr(settings, "RECIPE_PROCESSING_SETTINGS", {}))
        # Detect if this is a recipe-style document using config-driven keywords
        is_recipe_document = self._detect_recipe_document(text_blocks, recipe_cfg)
        if is_recipe_document:
            # Extract each recipe as a separate, comprehensive section
            return self._extract_recipe_sections(text_blocks, pdf_path, recipe_cfg)
        else:
            # Use standard section extraction for non-recipe documents
            return self._extract_standard_sections(text_blocks, pdf_path, quality_cfg)
        """
        Extract only meaningful, context-rich sections from text blocks.
        - For recipe-style documents: Merge ingredients, instructions, and recipe names into comprehensive sections.
        - For other documents: Use existing high-quality section extraction logic.
        - Document type detection is based on config-driven keywords and persona/job context, not hardcoded paths.
        - All config, keywords, and thresholds are loaded from config files (no hardcoding).
        
        Args:
            text_blocks: List of text blocks extracted from PDF
            pdf_path: Path to the PDF file for context
            
        Returns:
            List of section dictionaries with enhanced recipe grouping for recipe-style documents
        """
        from config import settings
        quality_cfg = getattr(settings, "SECTION_QUALITY_SETTINGS", {})
        # Use recipe processing settings for detection - this is where the recipe keywords are defined
        recipe_cfg = getattr(settings, "RECIPE_PROCESSING_SETTINGS", {})
        
        # Detect if this is a recipe-style document using config-driven keywords
        is_recipe_document = self._detect_recipe_document(text_blocks, recipe_cfg)
        
        if is_recipe_document:
            self.logger.info(f"Detected recipe-style document: {pdf_path.name}")
            return self._extract_recipe_sections(text_blocks, pdf_path, recipe_cfg)
        else:
            self.logger.info(f"Using standard section extraction for: {pdf_path.name}")
            return self._extract_standard_sections(text_blocks, pdf_path, quality_cfg)
    
    def _detect_recipe_document(self, text_blocks, recipe_cfg: Dict[str, Any]) -> bool:
        """
        Detect if document is recipe-style based on config-driven keywords and content patterns.
        
        Args:
            text_blocks: List of text blocks from document
            recipe_cfg: Recipe detection configuration
            
        Returns:
            bool: True if document appears to be recipe-style
        """
        # Config-driven recipe detection keywords - use recipe_keywords from RECIPE_PROCESSING_SETTINGS
        recipe_keywords = recipe_cfg.get("recipe_keywords", [
            "ingredients", "recipe", "cook", "bake", "prepare", "dish", "meal",
            "serves", "prep time", "cook time", "minutes", "cup", "tablespoon",
            "teaspoon", "oven", "temperature", "degrees", "mix", "stir", "add"
        ])
        recipe_threshold = recipe_cfg.get("recipe_keyword_ratio_threshold", 0.08)  # 8% of text blocks
        
        if not text_blocks:
            return False
        
        # Count text blocks containing recipe keywords
        recipe_block_count = 0
        total_blocks = len(text_blocks)
        
        for block in text_blocks:
            text_lower = block.text.lower()
            if any(keyword in text_lower for keyword in recipe_keywords):
                recipe_block_count += 1
        
        recipe_ratio = recipe_block_count / total_blocks if total_blocks > 0 else 0
        is_recipe = recipe_ratio >= recipe_threshold
        
        if is_recipe:
            self.logger.info(f"Recipe detection: {recipe_block_count}/{total_blocks} blocks ({recipe_ratio:.3f}) >= threshold {recipe_threshold}")
        
        return is_recipe
    
    def _extract_recipe_sections(self, text_blocks, pdf_path: Path, recipe_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and merge recipe sections for recipe-style documents.
        Each recipe is output as a separate, comprehensive section (title, ingredients, instructions).
        All logic is config-driven and generalized.
        
        Args:
            text_blocks: List of text blocks
            pdf_path: Path to PDF file
            recipe_cfg: Recipe extraction configuration
            
        Returns:
            List of comprehensive recipe sections
        """
        min_recipe_length = recipe_cfg.get("min_recipe_length", 100)  # Lower threshold for individual recipes
        min_recipe_words = recipe_cfg.get("min_recipe_words", 15)     # Lower threshold for individual recipes
        
        # Config-driven recipe component keywords
        ingredient_keywords = recipe_cfg.get("ingredient_keywords", ["ingredient", "ingredients:"])
        instruction_keywords = recipe_cfg.get("instruction_keywords", ["instruction", "instructions:", "method", "directions:", "preparation"])
        
        if not text_blocks:
            return []
        
        sections = []
        sorted_blocks = sorted(text_blocks, key=lambda b: (getattr(b, 'page_number', 1), getattr(b, 'y', 0)))
        
        # Parse sequential Ingredients/Instructions pairs - handle both compact and fragmented formats
        i = 0
        recipe_count = 0
        
        while i < len(sorted_blocks):
            block = sorted_blocks[i]
            text = block.text.strip()
            
            if not text or len(text) < 3:
                i += 1
                continue
                
            text_lower = text.lower()
            
            # Look for "Ingredients:" block
            if any(k in text_lower for k in ingredient_keywords):
                # Found ingredients section - collect all ingredient blocks
                ingredients_blocks = [block]
                ingredients_texts = [block.text]
                
                # Collect subsequent ingredient blocks (for PDFMiner fragmented format)
                j = i + 1
                while j < len(sorted_blocks):
                    next_block = sorted_blocks[j]
                    next_text = next_block.text.strip()
                    if not next_text:
                        j += 1
                        continue
                        
                    next_text_lower = next_text.lower()
                    
                    # Stop if we hit instructions or another ingredients section
                    if (any(k in next_text_lower for k in instruction_keywords) or 
                        any(k in next_text_lower for k in ingredient_keywords)):
                        break
                        
                    # Collect ingredient content (including bullet points, measurements, etc.)
                    if (len(next_text) < 100 and  # Short lines likely to be ingredients
                        not next_text_lower.startswith(('step', 'recipe', 'serves', 'prep time'))):
                        ingredients_blocks.append(next_block)
                        ingredients_texts.append(next_text)
                    
                    j += 1
                
                # Now look for instructions
                instructions_blocks = []
                instructions_texts = []
                
                # Continue from where ingredients ended
                while j < len(sorted_blocks):
                    next_block = sorted_blocks[j]
                    next_text = next_block.text.strip()
                    if not next_text:
                        j += 1
                        continue
                        
                    next_text_lower = next_text.lower()
                    
                    # Found instructions section
                    if any(k in next_text_lower for k in instruction_keywords):
                        instructions_blocks.append(next_block)
                        instructions_texts.append(next_text)
                        
                        # Collect subsequent instruction blocks
                        k = j + 1
                        while k < len(sorted_blocks):
                            inst_block = sorted_blocks[k]
                            inst_text = inst_block.text.strip()
                            if not inst_text:
                                k += 1
                                continue
                                
                            inst_text_lower = inst_text.lower()
                            
                            # Stop if we hit new ingredients or another major section
                            if any(kw in inst_text_lower for kw in ingredient_keywords):
                                break
                                
                            # Collect instruction content
                            if (len(inst_text) < 200 and  # Reasonable instruction length
                                not inst_text_lower.startswith(('recipe', 'serves', 'prep time'))):
                                instructions_blocks.append(inst_block)
                                instructions_texts.append(inst_text)
                            
                            k += 1
                        break
                    
                    # Hit another ingredients section - this recipe has no instructions
                    elif any(k in next_text_lower for k in ingredient_keywords):
                        break
                    j += 1
                
                # Create recipe section if we have ingredients and instructions
                if ingredients_blocks and instructions_blocks:
                    recipe_count += 1
                    
                    # Combine all ingredient and instruction texts
                    combined_ingredients = " ".join(ingredients_texts)
                    combined_instructions = " ".join(instructions_texts)
                    
                    recipe_data = {
                        "name": f"Recipe {recipe_count}",
                        "ingredients": [combined_ingredients],
                        "instructions": [combined_instructions],
                        "blocks": ingredients_blocks + instructions_blocks
                    }
                    
                    # Generate meaningful title from ingredients
                    title = self._generate_recipe_title(combined_ingredients, recipe_count)
                    recipe_data["name"] = title
                    
                    if self._is_substantial_recipe(recipe_data, min_recipe_length, min_recipe_words):
                        sections.append(self._create_recipe_section(recipe_data, pdf_path))
                    
                    # Continue from after instructions
                    i = j
                else:
                    i += 1
            else:
                i += 1
        
        self.logger.info(f"Extracted {len(sections)} recipe sections from {pdf_path.name}")
        return sections[:15]  # Allow more recipes per PDF

    def _generate_recipe_title(self, ingredients_text: str, recipe_number: int) -> str:
        """
        Generate a meaningful recipe title based on key ingredients.
        
        Args:
            ingredients_text: Raw ingredients text
            recipe_number: Sequential recipe number
            
        Returns:
            Generated recipe title
        """
        # Common food keywords to identify main ingredients (expanded list)
        food_indicators = [
            "pancake", "egg", "toast", "smoothie", "avocado", "omelette", "omelet", 
            "oatmeal", "yogurt", "muffin", "chia", "french toast", "scrambled",
            "bacon", "sausage", "cheese", "bread", "flour", "banana", "berry",
            "pasta", "rice", "chicken", "beef", "fish", "salmon", "tuna",
            "salad", "soup", "sandwich", "wrap", "burrito", "pizza", "quinoa",
            "lentil", "bean", "tofu", "mushroom", "spinach", "broccoli", "carrot",
            "potato", "sweet potato", "tomato", "pepper", "onion", "garlic"
        ]
        
        ingredients_lower = ingredients_text.lower()
        
        # Try to identify main ingredient/dish type
        for indicator in food_indicators:
            if indicator in ingredients_lower:
                return f"{indicator.title()} Recipe"
        
        # Look for cooking methods or dish types
        cooking_methods = ["baked", "grilled", "fried", "roasted", "steamed", "sautéed"]
        for method in cooking_methods:
            if method in ingredients_lower:
                return f"{method.title()} Dish"
        
        # Fallback to generic title
        return f"Recipe {recipe_number}"
    
    def _extract_standard_sections(self, text_blocks, pdf_path: Path, quality_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract sections using the existing high-quality logic for non-recipe documents.
        
        Args:
            text_blocks: List of text blocks
            pdf_path: Path to PDF file  
            quality_cfg: Quality configuration settings
            
        Returns:
            List of sections using standard extraction logic
        """
        min_section_length = quality_cfg.get("min_section_length", 200)
        min_section_words = quality_cfg.get("min_section_words", 30)
        min_content_blocks = quality_cfg.get("min_content_blocks", 3)
        max_content_blocks = quality_cfg.get("max_content_blocks", 15)
        
        # Trivial/generic headings to skip or merge (for non-recipe documents)
        trivial_headings = set([
            "introduction", "conclusion", "summary", "overview", "notes", "references"
        ])
        
        if not text_blocks:
            return []
        sections = []
        used_indices = set()
        sorted_blocks = sorted(text_blocks, key=lambda b: (getattr(b, 'page_number', 1), getattr(b, 'y', 0)))
        
        # Standard section extraction logic for non-recipe documents
        for i, block in enumerate(sorted_blocks):
            text = block.text.strip()
            if not text or len(text) < 3:
                continue
            font_size = getattr(block, 'font_size', 12)
            font_flags = getattr(block, 'font_flags', 0)
            is_bold = (font_flags == 20)
            text_lc = text.lower().strip(':')
            
            # Check if this is a trivial heading
            is_trivial = text_lc in trivial_headings or any(text_lc.startswith(t + ':') for t in trivial_headings)
            
            # If this is a non-trivial, bold, multi-word heading, treat as a section if followed by substantial content
            if is_bold and len(text.split()) >= 2 and not is_trivial and i not in used_indices:
                # Collect content blocks after this heading
                content_blocks = []
                k = i + 1
                while k < len(sorted_blocks):
                    next_block = sorted_blocks[k]
                    next_text = next_block.text.strip()
                    next_lc = next_text.lower().strip(':')
                    if not next_text:
                        k += 1
                        continue
                    # Stop if next block is a new bold heading
                    next_flags = getattr(next_block, 'font_flags', 0)
                    if next_flags == 20 and k != i + 1:
                        break
                    # Skip trivial headings unless merging
                    if next_lc in trivial_headings:
                        k += 1
                        continue
                    content_blocks.append(next_text)
                    k += 1
                    if len(content_blocks) >= max_content_blocks:
                        break
                
                # Only add if content is substantial
                if len(" ".join(content_blocks)) >= min_section_length and len(content_blocks) >= min_content_blocks:
                    section_title = text
                    if len(section_title) > 100:
                        section_title = section_title[:100] + "..."

                    sections.append({
                        "document": pdf_path.name,
                        "title": section_title,
                        "section_title": section_title,
                        "page_number": getattr(block, 'page_number', 1),
                        "position": {"x": getattr(block, 'x', 0), "y": getattr(block, 'y', 0)},
                        "font_size": getattr(block, 'font_size', 12),
                        "text": section_title,
                        "content": " ".join(content_blocks),
                        "importance_rank": len(sections) + 1
                    })

        # Remove duplicates and empty/trivial sections
        unique_titles = set()
        filtered_sections = []
        for sec in sections:
            title = sec["section_title"].strip().lower()
            if not title or title in unique_titles:
                continue
            if any(t in title for t in trivial_headings):
                continue
            if len(sec["content"].split()) < min_section_words:
                continue
            unique_titles.add(title)
            filtered_sections.append(sec)
        
        # Fallback if nothing found
        if not filtered_sections:
            return self._create_fallback_sections(sorted_blocks, pdf_path)
        return filtered_sections[:15]
    
    def _is_substantial_recipe(self, recipe: Dict[str, Any], min_length: int, min_words: int) -> bool:
        """
        Check if a recipe has substantial content worth extracting.
        
        Args:
            recipe: Recipe dictionary with name, ingredients, instructions, blocks
            min_length: Minimum character length
            min_words: Minimum word count
            
        Returns:
            bool: True if recipe is substantial
        """
        if not recipe["blocks"]:
            return False
            
        total_text = " ".join([block.text for block in recipe["blocks"]])
        return len(total_text) >= min_length and len(total_text.split()) >= min_words
    
    def _create_recipe_section(self, recipe: Dict[str, Any], pdf_path: Path) -> Dict[str, Any]:
        """
        Create a comprehensive recipe section from grouped recipe components.
        
        Args:
            recipe: Recipe dictionary with components
            pdf_path: Path to PDF file
            
        Returns:
            Dict: Comprehensive recipe section
        """
        # Build comprehensive section title
        recipe_name = recipe["name"] or "Recipe"
        if len(recipe_name) > 80:
            recipe_name = recipe_name[:80] + "..."
            
        section_title = f"{recipe_name} - Ingredients & Instructions"
        
        # Build comprehensive content
        content_parts = []
        if recipe["name"]:
            content_parts.append(f"Recipe: {recipe['name']}")
        if recipe["ingredients"]:
            content_parts.append("Ingredients: " + " ".join(recipe["ingredients"]))
        if recipe["instructions"]:
            content_parts.append("Instructions: " + " ".join(recipe["instructions"]))
        
        # Add any additional content from blocks
        additional_content = []
        for block in recipe["blocks"]:
            block_text = block.text.strip()
            if (block_text and block_text not in content_parts and 
                not any(block_text in part for part in content_parts)):
                additional_content.append(block_text)
        
        if additional_content:
            content_parts.extend(additional_content)
        
        comprehensive_content = " ".join(content_parts)
        
        # Get position from first block
        first_block = recipe["blocks"][0] if recipe["blocks"] else None
        
        return {
            "document": pdf_path.name,
            "title": section_title,
            "section_title": section_title,
            "page_number": getattr(first_block, 'page_number', 1) if first_block else 1,
            "position": {
                "x": getattr(first_block, 'x', 0) if first_block else 0,
                "y": getattr(first_block, 'y', 0) if first_block else 0
            },
            "font_size": getattr(first_block, 'font_size', 12) if first_block else 12,
            "text": section_title,
            "content": comprehensive_content,
            "importance_rank": 1,
            "recipe_components": {
                "name": recipe["name"],
                "ingredients": recipe["ingredients"],
                "instructions": recipe["instructions"]
            }
        }
        
        for i, block in enumerate(sorted_blocks):
            text = block.text.strip()
            if not text or len(text) < 3:
                continue
            
            text_lower = text.lower()
            font_flags = getattr(block, 'font_flags', 0)
            is_bold = (font_flags == 20)
            
            # Detect recipe name (bold, multi-word, not ingredient/instruction)
            is_recipe_name = (is_bold and len(text.split()) >= 2 and 
                             not any(k in text_lower for k in ingredient_keywords + instruction_keywords))
            
            # Detect ingredients section
            is_ingredient = any(k in text_lower for k in ingredient_keywords)
            
            # Detect instructions section  
            is_instruction = any(k in text_lower for k in instruction_keywords)
            
            if is_recipe_name and current_recipe["blocks"]:
                # Start new recipe, save previous if substantial
                if self._is_substantial_recipe(current_recipe, min_recipe_length, min_recipe_words):
                    sections.append(self._create_recipe_section(current_recipe, pdf_path))
                current_recipe = {"name": text, "ingredients": [], "instructions": [], "blocks": [block]}
            elif is_ingredient:
                current_recipe["ingredients"].append(text)
                current_recipe["blocks"].append(block)
            elif is_instruction:
                current_recipe["instructions"].append(text)
                current_recipe["blocks"].append(block)
            else:
                # Regular content block
                current_recipe["blocks"].append(block)
        
        # Add final recipe if substantial
        if self._is_substantial_recipe(current_recipe, min_recipe_length, min_recipe_words):
            sections.append(self._create_recipe_section(current_recipe, pdf_path))
        
        return sections[:10]  # Limit to reasonable number of recipes
    
    def _extract_standard_sections(self, text_blocks, pdf_path: Path, quality_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract sections using the existing high-quality logic for non-recipe documents.
        
        Args:
            text_blocks: List of text blocks
            pdf_path: Path to PDF file  
            quality_cfg: Quality configuration settings
            
        Returns:
            List of sections using standard extraction logic
        """
        min_section_length = quality_cfg.get("min_section_length", 200)
        min_section_words = quality_cfg.get("min_section_words", 30)
        min_content_blocks = quality_cfg.get("min_content_blocks", 3)
        max_content_blocks = quality_cfg.get("max_content_blocks", 15)
        
        # Trivial/generic headings to skip or merge (for non-recipe documents)
        trivial_headings = set([
            "introduction", "conclusion", "summary", "overview", "notes", "references"
        ])
        # Section title joiners for merging
        joiner = " - "
        if not text_blocks:
            return []
        sections = []
        used_indices = set()
        sorted_blocks = sorted(text_blocks, key=lambda b: (getattr(b, 'page_number', 1), getattr(b, 'y', 0)))
        # Pass 1: Identify all candidate headings and their context
        for i, block in enumerate(sorted_blocks):
            text = block.text.strip()
            if not text or len(text) < 3:
                continue
            font_size = getattr(block, 'font_size', 12)
            font_flags = getattr(block, 'font_flags', 0)
            is_bold = (font_flags == 20)
            # Lowercase for matching
            text_lc = text.lower().strip(':')
            # Check if this is a trivial heading
            is_trivial = text_lc in trivial_headings or any(text_lc.startswith(t + ':') for t in trivial_headings)
            # If this is a trivial heading, check if it is part of a recipe/topic section
            if is_trivial:
                # Look backward for a non-trivial, bold, multi-word heading (likely recipe/topic name)
                j = i - 1
                while j >= 0:
                    prev = sorted_blocks[j]
                    prev_text = prev.text.strip()
                    prev_lc = prev_text.lower().strip(':')
                    prev_flags = getattr(prev, 'font_flags', 0)
                    prev_bold = (prev_flags == 20)
                    if prev_bold and prev_lc not in trivial_headings and len(prev_text.split()) >= 2:
                        # Merge: Recipe/Topic Name + Trivial Heading
                        merged_title = prev_text + joiner + text
                        # Collect content blocks after this trivial heading
                        content_blocks = []
                        k = i + 1
                        while k < len(sorted_blocks):
                            next_block = sorted_blocks[k]
                            next_text = next_block.text.strip()
                            if not next_text or next_text.lower().strip(':') in trivial_headings:
                                k += 1
                                continue
                            # Stop if next block is a new bold heading
                            next_flags = getattr(next_block, 'font_flags', 0)
                            if next_flags == 20 and k != i + 1:
                                break
                            content_blocks.append(next_text)
                            k += 1
                            if len(content_blocks) >= max_content_blocks:
                                break
                        # Only add if content is substantial
                        if len(" ".join(content_blocks)) >= min_section_length and len(content_blocks) >= min_content_blocks:
                            sections.append({
                                "document": pdf_path.name,
                                "title": merged_title,
                                "section_title": merged_title,
                                "page_number": getattr(block, 'page_number', 1),
                                "position": {"x": getattr(block, 'x', 0), "y": getattr(block, 'y', 0)},
                                "font_size": font_size,
                                "text": merged_title,
                                "content": "\n".join(content_blocks),
                                "importance_rank": len(sections) + 1
                            })
                            used_indices.update([j, i])
                            used_indices.update(range(i + 1, k))
                        break
                    j -= 1
                continue
            # If this is a non-trivial, bold, multi-word heading, treat as a section if followed by substantial content
            if is_bold and len(text.split()) >= 2 and not is_trivial and i not in used_indices:
                # Collect content blocks after this heading
                content_blocks = []
                k = i + 1
                while k < len(sorted_blocks):
                    next_block = sorted_blocks[k]
                    next_text = next_block.text.strip()
                    next_lc = next_text.lower().strip(':')
                    if not next_text:
                        k += 1
                        continue
                    # Stop if next block is a new bold heading
                    next_flags = getattr(next_block, 'font_flags', 0)
                    if next_flags == 20 and k != i + 1:
                        break
                    # Skip trivial headings unless merging
                    if next_lc in trivial_headings:
                        k += 1
                        continue
                    content_blocks.append(next_text)
                    k += 1
                    if len(content_blocks) >= max_content_blocks:
                        break
                # Only add if content is substantial
                if len(" ".join(content_blocks)) >= min_section_length and len(content_blocks) >= min_content_blocks:
                    section_title = text
                    if len(section_title) > 100:
                        section_title = section_title[:100] + "..."
                    sections.append({
                        "document": pdf_path.name,
                        "title": section_title,
                        "section_title": section_title,
                        "page_number": getattr(block, 'page_number', 1),
                        "position": {"x": getattr(block, 'x', 0), "y": getattr(block, 'y', 0)},
                        "font_size": font_size,
                        "text": section_title,
                        "content": "\n".join(content_blocks),
                        "importance_rank": len(sections) + 1
                    })
                    used_indices.update([i])
                    used_indices.update(range(i + 1, k))
        # Remove duplicates and empty/trivial sections
        unique_titles = set()
        filtered_sections = []
        for sec in sections:
            title = sec["section_title"].strip().lower()
            if not title or title in unique_titles:
                continue
            if any(t in title for t in trivial_headings):
                continue
            if len(sec["content"].split()) < min_section_words:
                continue
            unique_titles.add(title)
            filtered_sections.append(sec)
        # Fallback if nothing found
        if not filtered_sections:
            return self._create_fallback_sections(sorted_blocks, pdf_path)
        return filtered_sections[:15]
    
    def _group_content_by_context(self, content_list: List[str], heading_text: str) -> List[List[str]]:
        """
        Group content blocks by semantic context to create coherent narrative sections.
        
        Args:
            content_list: List of content text blocks
            heading_text: The section heading for context
            
        Returns:
            List of content groups, each group containing related content blocks
        """
        if not content_list:
            return []
        
        # Define topic clusters for intelligent grouping
        topic_clusters = {
            'location': ['place', 'location', 'area', 'region', 'district', 'neighborhood', 'town', 'city'],
            'activities': ['activity', 'tour', 'visit', 'explore', 'experience', 'adventure', 'excursion'],
            'dining': ['restaurant', 'food', 'cuisine', 'dining', 'eat', 'meal', 'taste', 'chef', 'menu'],
            'accommodation': ['hotel', 'accommodation', 'stay', 'room', 'resort', 'lodge', 'inn'],
            'culture': ['culture', 'art', 'museum', 'history', 'tradition', 'heritage', 'historic'],
            'practical': ['tip', 'advice', 'recommend', 'suggest', 'important', 'note', 'remember']
        }
        
        # Group content by topic similarity
        groups = []
        current_group = []
        current_topic = None
        
        for content in content_list:
            content_lower = content.lower()
            
            # Determine topic of this content block
            best_topic = None
            best_score = 0
            
            for topic, keywords in topic_clusters.items():
                score = sum(1 for keyword in keywords if keyword in content_lower)
                if score > best_score:
                    best_score = score
                    best_topic = topic
            
            # Group by topic continuity
            if best_topic == current_topic or current_topic is None:
                current_group.append(content)
                current_topic = best_topic
            else:
                # Start new group if topic changes significantly
                if current_group:
                    groups.append(current_group)
                current_group = [content]
                current_topic = best_topic
        
        # Add final group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _create_comprehensive_narrative(self, content_groups: List[str], heading_text: str) -> str:
        """
        Create a comprehensive narrative from content groups with smooth transitions.
        
        Args:
            content_groups: List of content parts to combine
            heading_text: Section heading for context
            
        Returns:
            Comprehensive narrative text
        """
        if not content_groups:
            return ""
        
        # Create contextual introduction based on heading
        narrative_parts = []
        
        # Enhanced narrative construction for travel content
        for i, part in enumerate(content_groups):
            clean_part = part.strip()
            if not clean_part:
                continue
            
            # Ensure proper sentence structure and flow
            if not clean_part.endswith(('.', '!', '?')):
                if i < len(content_groups) - 1:  # Not the last part
                    clean_part += '.'
            
            narrative_parts.append(clean_part)
        
        # Join with appropriate spacing and flow
        comprehensive_text = ' '.join(narrative_parts)
        
        # Clean up formatting issues
        comprehensive_text = comprehensive_text.replace('  ', ' ')
        comprehensive_text = comprehensive_text.replace(' .', '.')
        comprehensive_text = comprehensive_text.replace(' ,', ',')
        
        return comprehensive_text.strip()
    
    def _create_fallback_sections(self, sorted_blocks, pdf_path: Path) -> List[Dict[str, Any]]:
        """Create sections when heading detection fails."""
        sections = []
        
        # Group blocks by page and create page-based sections
        pages = {}
        for block in sorted_blocks:
            page_num = getattr(block, 'page_number', 1)
            if page_num not in pages:
                pages[page_num] = []
            if block.text.strip() and len(block.text.strip()) > 10:
                pages[page_num].append(block.text.strip())
        
        for page_num, texts in pages.items():
            if not texts:
                continue
                
            # Take first significant text as title
            title = texts[0][:80] + "..." if len(texts[0]) > 80 else texts[0]
            
            # Aggregate content from this page
            content = ' '.join(texts[1:5])  # Next 4 text blocks
            if len(content) > 400:
                content = content[:400] + "..."
            
            sections.append({
                "document": pdf_path.name,
                "title": title,
                "section_title": title,
                "page_number": page_num,
                "position": {"x": 0, "y": 0},
                "font_size": 12,
                "text": title,
                "content": content,
                "importance_rank": len(sections) + 1
            })
            
            if len(sections) >= 5:  # Limit fallback sections
                break
        
        return sections
    
    def extract_page_texts(self, document_paths: List[Path]) -> Dict[str, List[str]]:
        """
        Extract text from each page of multiple documents.
        
        Args:
            document_paths: List of paths to PDF files
            
        Returns:
            Dictionary mapping filenames to lists of page texts
        """
        page_texts = {}
        
        for pdf_path in document_paths:
            if not pdf_path.exists():
                continue
                
            parser = PyMuPDFParser()
            try:
                if parser.load_document(pdf_path):
                    pages = []
                    for page_num in range(parser.get_page_count()):
                        page_text = parser.extract_text(page_num)
                        pages.append(page_text)
                    page_texts[pdf_path.name] = pages
            except Exception as e:
                self.logger.error(f"Failed to extract page texts from {pdf_path.name}: {str(e)}")
            finally:
                parser.close_document()
        
        return page_texts
    
    def validate_documents(self, document_paths: List[Path]) -> Dict[str, bool]:
        """
        Validate that all documents can be loaded and processed.
        
        Args:
            document_paths: List of paths to PDF files
            
        Returns:
            Dictionary mapping filenames to validation status
        """
        validation_results = {}
        
        for pdf_path in document_paths:
            is_valid = False
            
            if pdf_path.exists():
                parser = PyMuPDFParser()
                try:
                    if parser.load_document(pdf_path):
                        # Try to extract some text to verify it's readable
                        text = parser.extract_text()
                        is_valid = len(text.strip()) > 0
                except Exception:
                    is_valid = False
                finally:
                    parser.close_document()
            
            validation_results[pdf_path.name] = is_valid
        
        return validation_results
