"""
Configuration management for Adobe Hackathon Round 1B.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
TESTS_DIR = PROJECT_ROOT / "tests"
CONFIG_DIR = PROJECT_ROOT / "config"

# Input/Output paths
INPUT_DIR = Path("/app/input") if os.path.exists("/app") else PROJECT_ROOT / "input"
OUTPUT_DIR = Path("/app/output") if os.path.exists("/app") else PROJECT_ROOT / "output"

# Performance constraints - RESTORED QUALITY with PRE-WARMED MODEL
MAX_PROCESSING_TIME = 25  # seconds - achievable with pre-warmed models
MAX_MODEL_SIZE = 1024  # MB - restored for quality
MAX_MEMORY_USAGE = 16384  # MB (16GB) - restored for quality

# Processing settings - QUALITY RESTORED
BATCH_SIZE = 32  # restored for quality balance
MAX_SEQUENCE_LENGTH = 512  # restored for quality
EMBEDDING_CACHE_SIZE = 5000  # increased cache for speed

# Quality optimization settings for Round 1B - QUALITY RESTORED
SECTION_QUALITY_SETTINGS = {
    # Minimum thresholds for meaningful sections - RESTORED FOR QUALITY
    "min_section_length": 200,  # restored from 100
    "min_section_words": 30,     # restored from 15
    "min_content_blocks": 3,     # restored from 2
    
    # Content aggregation settings - OPTIMIZED
    "max_content_blocks": 10,    # reduced from 15
    "target_section_length": 400, # reduced from 600
    "max_section_length": 800,   # reduced from 1200
    
    # Quality scoring weights - SIMPLIFIED
    "content_diversity_weight": 0.2,   # reduced computation
    "informativeness_weight": 0.5,    # focused on key metric
    "context_relevance_weight": 0.3,
    
    # Penalty thresholds for low-quality sections - RELAXED
    "generic_title_penalty": -0.3,    # less strict
    "fragment_penalty": -0.2,         # less strict
    "short_content_penalty": -0.1     # less strict
}

# Recipe document detection and processing settings
RECIPE_PROCESSING_SETTINGS = {
    # Keywords for detecting recipe-style content
    "recipe_keywords": [
        "ingredients", "recipe", "cook", "bake", "prepare", "dish", "meal",
        "serves", "prep time", "cook time", "minutes", "cup", "tablespoon",
        "teaspoon", "oven", "temperature", "degrees", "mix", "stir", "add"
    ],
    
    # Minimum ratio of recipe keywords to total words for detection
    "recipe_keyword_ratio_threshold": 0.08,
    
    # Recipe section merging settings
    "min_recipe_component_length": 50,
    "recipe_name_max_length": 100,
    "ingredients_section_min_items": 3,
    "instructions_section_min_steps": 2,
    
    # Recipe extraction settings
    "min_recipe_length": 100,
    "min_recipe_words": 15,
    "ingredient_keywords": ["ingredient", "ingredients:"],
    "instruction_keywords": ["instruction", "instructions:", "method", "directions:", "preparation"]
}

# Subsection quality settings for different document types - SPEED OPTIMIZED
SUBSECTION_QUALITY_SETTINGS = {
    # General passage quality thresholds - REDUCED FOR SPEED
    "min_passage_length": 100,        # reduced from 150
    "target_passage_length": 300,     # reduced from 400
    "max_passage_length": 600,        # reduced from 800
    "min_information_density": 2,     # reduced from 3
    "persona_weight": 0.4,
    "job_weight": 0.6,
    "deduplication_threshold": 0.88,  # slightly relaxed
    "max_passages_per_section": 2,    # reduced from 3
    "max_global_passages": 20,        # reduced from 30
    "min_unique_tokens": 8,           # reduced from 12
    "min_sentence_count": 1,          # reduced from 2
    
    # Recipe-specific passage settings - OPTIMIZED
    "recipe_passage_settings": {
        "min_passage_length": 80,     # reduced from 100
        "target_passage_length": 250, # reduced from 300
        "max_passage_length": 500,    # reduced from 600
        "merge_ingredients_instructions": True,
        "require_both_ingredients_instructions": True,
        "comprehensive_recipe_format": True
    }
}

# Semantic grouping settings for passage creation
SEMANTIC_GROUPING_SETTINGS = {
    # Keywords for identifying recipe components
    "ingredient_keywords": ["ingredient", "ingredients:", "items needed", "you will need"],
    "instruction_keywords": ["instruction", "instructions:", "method", "directions:", "steps:", "preparation"],
    
    # Text block delimiters for splitting content
    "block_delimiters": ["  ", "\n\n", ". ", "! ", "? "],
    
    # Recipe grouping thresholds
    "min_recipe_components": 2,  # Need both ingredients and instructions
    "max_recipe_length": 1000,   # Maximum characters for a single recipe passage
    "ingredient_instruction_gap_threshold": 200  # Max gap between ingredients and instructions
}

# Passage quality assessment for different content types
PASSAGE_QUALITY_SETTINGS = {
    # Travel content value indicators
    "travel_value_indicators": [
        "explore", "visit", "discover", "experience", "enjoy", "see", "try",
        "beautiful", "stunning", "perfect", "ideal", "popular", "famous",
        "recommend", "suggest", "located", "situated", "features", "offers"
    ],
    
    # Recipe/food content value indicators
    "recipe_value_indicators": [
        "prepare", "cook", "bake", "mix", "serve", "recipe", "dish", "meal",
        "flavor", "taste", "delicious", "nutritious", "healthy", "fresh",
        "ingredients", "season", "garnish", "texture", "aroma"
    ],
    
    # Technical/business content value indicators
    "technical_value_indicators": [
        "feature", "function", "process", "method", "step", "procedure",
        "important", "required", "complete", "configure", "enable", "access",
        "click", "select", "choose", "open", "create", "save", "export"
    ],
    
    # Generic fragments to exclude from quality passages
    "generic_exclude_fragments": [
        "see above", "see below", "as mentioned", "refer to", "page",
        "figure", "table", "chart", "note:", "tip:", "warning:",
        "step 1", "step 2", "step 3", "item 1", "item 2"
    ]
}

# Persona and job filtering settings for content appropriateness
PERSONA_JOB_FILTERING = {
    # Dietary restriction keywords for filtering
    "vegetarian_indicators": ["vegetarian", "vegan", "plant-based", "veggie"],
    "gluten_free_indicators": ["gluten-free", "gluten free", "celiac", "wheat-free"],
    
    # Content exclusion keywords for specific dietary needs
    "non_vegetarian_exclude": [
        # Primary meat keywords
        "meat", "beef", "pork", "chicken", "fish", "seafood", "lamb",
        "turkey", "bacon", "sausage", "ham", "salmon", "tuna", "shrimp",
        "crab", "lobster", "scallops", "mussels", "oysters", "clams",
        
        # Poultry and derivatives
        "duck", "goose", "quail", "poultry", "wing", "drumstick", 
        "breast", "thigh", "giblets", "liver", "kidney",
        
        # Red meat variations
        "steak", "roast", "ground beef", "mince", "veal", "venison",
        "bison", "elk", "rabbit", "goat", "mutton",
        
        # Processed meats
        "pepperoni", "salami", "chorizo", "prosciutto", "pastrami",
        "bologna", "hot dog", "bratwurst", "kielbasa", "mortadella",
        
        # Seafood and marine life
        "anchovy", "sardine", "mackerel", "cod", "halibut", "bass",
        "trout", "catfish", "tilapia", "swordfish", "eel", "octopus",
        "squid", "calamari", "caviar", "roe",
        
        # Animal-derived ingredients
        "gelatin", "lard", "tallow", "suet", "bone marrow", "blood",
        "rennet", "isinglass", "cochineal", "carmine",
        
        # Cooking methods that indicate meat
        "grilled chicken", "fried fish", "roasted beef", "smoked salmon",
        "barbecued", "broiled fish", "pan-seared", "braised meat"
    ],
    "gluten_exclude": [
        "wheat", "flour", "bread", "pasta", "gluten", "barley", "rye",
        "beer", "malt", "cereal", "crackers", "cookies"
    ],
    
    # Professional context filtering
    "corporate_inappropriate": [
        "alcohol", "wine", "beer", "cocktail", "drink", "bar", "pub",
        "party", "celebration", "festive", "holiday specific"
    ],
    
    # Health-conscious filtering
    "unhealthy_exclude": [
        "fried", "deep-fried", "high-fat", "processed", "artificial",
        "preservatives", "additives", "sugar-heavy", "calorie-dense"
    ]
}

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Docker settings
DOCKER_INPUT_PATH = "/app/input"
DOCKER_OUTPUT_PATH = "/app/output"
DOCKER_MODELS_PATH = "/app/models"

def load_config():
    """
    Load configuration settings as a dictionary.
    
    Returns:
        dict: Configuration dictionary with all settings
    """
    return {
        'input_dir': INPUT_DIR,
        'output_dir': OUTPUT_DIR,
        'input_json': 'challenge1b_input.json',
        'output_json': 'challenge1b_output.json',
        'max_processing_time': MAX_PROCESSING_TIME,
        'batch_size': BATCH_SIZE,
        'max_sequence_length': MAX_SEQUENCE_LENGTH,
        'embedding_cache_size': EMBEDDING_CACHE_SIZE,
        'log_level': LOG_LEVEL,
        'log_format': LOG_FORMAT,
        'models_dir': MODELS_DIR,
        'cache_max_size': 1000,
        'cache_persistence_path': None,
        
        # Quality settings
        'SECTION_QUALITY_SETTINGS': SECTION_QUALITY_SETTINGS,
        'SUBSECTION_QUALITY_SETTINGS': SUBSECTION_QUALITY_SETTINGS,
        'SEMANTIC_GROUPING_SETTINGS': SEMANTIC_GROUPING_SETTINGS,
        'RECIPE_PROCESSING_SETTINGS': RECIPE_PROCESSING_SETTINGS,
        'PASSAGE_QUALITY_SETTINGS': PASSAGE_QUALITY_SETTINGS,
        'PERSONA_JOB_FILTERING': PERSONA_JOB_FILTERING,
    }
