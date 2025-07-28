"""
ML model configurations for Adobe Hackathon Round 1B.
"""

# Model path constants - using model names for sentence-transformers (cached automatically)
DEFAULT_EMBEDDING_MODEL_PATH = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LANGUAGE_DETECTION_MODEL_PATH = "models/language_detection"
DEFAULT_OCR_MODEL_PATH = "models/tesseract_models"

# Primary multilingual model configuration - SPEED OPTIMIZED
PRIMARY_MODEL_CONFIG = {
    "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "size_mb": 278,
    "embedding_dimensions": 384,
    "max_sequence_length": 64,    # REDUCED from 128 for speed
    "languages_supported": 50,
    "performance_cpu": 150,       # INCREASED throughput expectation
    "cache_embeddings": True,
    "batch_size": 64,            # INCREASED from 32 for efficiency
    "use_fast_tokenizer": True,  # ADDED for speed
    "device_map": "auto"         # ADDED for optimization
}

# Fallback model configuration
FALLBACK_MODEL_CONFIG = {
    "name": "TF-IDF + Cosine Similarity",
    "size_mb": 50,
    "type": "classical_ml",
    "library": "scikit-learn",
    "languages_supported": "universal",
    "performance_cpu": 1000,  # sentences per second
    "use_for_emergency": True
}

# Language detection model
LANGUAGE_DETECTION_CONFIG = {
    "name": "langdetect",
    "size_mb": 5,
    "supported_languages": [
        "en", "es", "fr", "de", "it", "pt", "nl", "ru",
        "ja", "ko", "zh", "ar", "hi", "th", "vi", "tr"
    ],
    "confidence_threshold": 0.8
}

# Model size budget allocation (≤1GB total)
MODEL_BUDGET = {
    "primary_transformer": 278,  # MB
    "language_detection": 50,    # MB
    "classical_ml": 100,         # MB
    "ocr_language_packs": 200,   # MB
    "supporting_libraries": 300, # MB
    "runtime_buffer": 72,        # MB
    "total": 1000               # MB (exactly 1GB)
}

# OCR language configurations
OCR_CONFIG = {
    "tesseract_languages": {
        "ja": "jpn",
        "zh": "chi_sim+chi_tra",
        "ar": "ara",
        "hi": "hin",
        "ko": "kor",
        "th": "tha",
        "ru": "rus",
        "fr": "fra",
        "de": "deu",
        "es": "spa",
        "it": "ita"
    },
    "confidence_threshold": 60,
    "preprocessing": {
        "resize_factor": 2,
        "noise_removal": True,
        "contrast_enhancement": True
    }
}

# Model loading settings
MODEL_LOADING = {
    "preload_on_startup": True,
    "cache_directory": "/root/.cache",
    "download_timeout": 300,  # seconds
    "max_retries": 3,
    "verify_integrity": True
}
