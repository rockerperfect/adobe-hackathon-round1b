"""
Multilingual support configuration for Adobe Hackathon Round 1B.
"""

# Supported languages with metadata
SUPPORTED_LANGUAGES = {
    # European Languages
    "en": {"name": "English", "family": "Germanic", "rtl": False, "complex_script": False},
    "es": {"name": "Spanish", "family": "Romance", "rtl": False, "complex_script": False},
    "fr": {"name": "French", "family": "Romance", "rtl": False, "complex_script": False},
    "de": {"name": "German", "family": "Germanic", "rtl": False, "complex_script": False},
    "it": {"name": "Italian", "family": "Romance", "rtl": False, "complex_script": False},
    "pt": {"name": "Portuguese", "family": "Romance", "rtl": False, "complex_script": False},
    "nl": {"name": "Dutch", "family": "Germanic", "rtl": False, "complex_script": False},
    "ru": {"name": "Russian", "family": "Slavic", "rtl": False, "complex_script": False},
    
    # Asian Languages
    "ja": {"name": "Japanese", "family": "Japonic", "rtl": False, "complex_script": True},
    "ko": {"name": "Korean", "family": "Koreanic", "rtl": False, "complex_script": True},
    "zh": {"name": "Chinese", "family": "Sino-Tibetan", "rtl": False, "complex_script": True},
    "hi": {"name": "Hindi", "family": "Indo-European", "rtl": False, "complex_script": True},
    "th": {"name": "Thai", "family": "Tai-Kadai", "rtl": False, "complex_script": True},
    "vi": {"name": "Vietnamese", "family": "Austroasiatic", "rtl": False, "complex_script": False},
    
    # Middle Eastern Languages
    "ar": {"name": "Arabic", "family": "Semitic", "rtl": True, "complex_script": True},
    "he": {"name": "Hebrew", "family": "Semitic", "rtl": True, "complex_script": True},
    "fa": {"name": "Persian", "family": "Indo-European", "rtl": True, "complex_script": True},
    
    # Other Languages
    "tr": {"name": "Turkish", "family": "Turkic", "rtl": False, "complex_script": False},
    "pl": {"name": "Polish", "family": "Slavic", "rtl": False, "complex_script": False},
    "sv": {"name": "Swedish", "family": "Germanic", "rtl": False, "complex_script": False},
    "no": {"name": "Norwegian", "family": "Germanic", "rtl": False, "complex_script": False},
    "da": {"name": "Danish", "family": "Germanic", "rtl": False, "complex_script": False}
}

# Language-specific preprocessing rules
PREPROCESSING_RULES = {
    # Asian languages - handle spacing and punctuation
    "asian": {
        "languages": ["ja", "zh", "ko"],
        "rules": {
            "normalize_spacing": True,
            "handle_mixed_punctuation": True,
            "character_normalization": True,
            "word_segmentation": False  # These languages don't use spaces
        }
    },
    
    # Right-to-left languages
    "rtl": {
        "languages": ["ar", "he", "fa"],
        "rules": {
            "direction_handling": True,
            "punctuation_normalization": True,
            "diacritic_handling": True,
            "number_handling": True  # Numbers are LTR in RTL text
        }
    },
    
    # Latin-based languages
    "latin": {
        "languages": ["en", "es", "fr", "de", "it", "pt", "nl", "ru", "tr", "pl", "sv", "no", "da"],
        "rules": {
            "case_normalization": False,  # Preserve case for semantic meaning
            "punctuation_standardization": True,
            "whitespace_normalization": True,
            "special_character_removal": False  # Keep accents and special chars
        }
    },
    
    # Complex script languages
    "complex": {
        "languages": ["hi", "th", "vi"],
        "rules": {
            "script_normalization": True,
            "compound_character_handling": True,
            "tone_mark_preservation": True
        }
    }
}

# Text normalization patterns
NORMALIZATION_PATTERNS = {
    "whitespace": {
        "pattern": r"\s+",
        "replacement": " ",
        "description": "Normalize multiple whitespace to single space"
    },
    "punctuation": {
        "pattern": r"[^\w\s\.,!?;:()\-]",
        "replacement": "",
        "description": "Remove non-standard punctuation while preserving basic ones"
    },
    "mixed_punctuation": {
        "asian_periods": {"。": ". ", "、": ", "},
        "arabic_punctuation": {"؟": "?", "،": ","},
        "description": "Convert language-specific punctuation to standard forms"
    }
}

# Language detection settings
LANGUAGE_DETECTION = {
    "min_text_length": 50,  # Minimum characters for reliable detection
    "confidence_threshold": 0.8,
    "fallback_language": "en",
    "multi_language_threshold": 0.3,  # If multiple languages detected
    "sample_size": 1000  # Characters to use for detection
}

# Multilingual model compatibility
MODEL_LANGUAGE_SUPPORT = {
    "primary_model": {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
        "shared_semantic_space": True,
        "cross_lingual_similarity": True
    },
    "fallback_model": {
        "model_name": "TF-IDF",
        "supported_languages": "universal",
        "language_agnostic": True
    }
}
