#!/usr/bin/env python3
"""
Model Pre-warming Script for Adobe Hackathon Pipeline
Loads and caches models BEFORE processing to eliminate 55s model loading time

This script should be run ONCE at container startup to pre-warm models.
Subsequent collection processing will be 20-30 seconds instead of 75+ seconds.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nlp.model_singleton import ModelSingleton
from src.nlp.embeddings import EmbeddingEngine
from config import model_config

def setup_logging():
    """Setup logging for pre-warming."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def pre_warm_models():
    """Pre-warm all required models to eliminate loading time during processing."""
    logger = setup_logging()
    
    logger.info("🔥 Starting model pre-warming for Adobe Hackathon Pipeline...")
    start_time = time.time()
    
    try:
        # Pre-warm singleton model
        logger.info("Pre-warming sentence transformer model...")
        singleton = ModelSingleton()
        model = singleton.get_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        logger.info(f"✅ Singleton model pre-warmed successfully")
        
        # Pre-warm embedding engines (4 instances as used in main pipeline)
        logger.info("Pre-warming embedding engines...")
        for i in range(4):
            engine = EmbeddingEngine()
            # Test with a small embedding to fully initialize
            test_embedding = engine.embed_text("test")
            logger.info(f"✅ Embedding engine {i+1}/4 pre-warmed")
        
        # Pre-warm with typical workload
        logger.info("Pre-warming with typical workload...")
        test_texts = [
            "Travel planning for South of France",
            "HR forms and document management", 
            "Vegetarian recipe for corporate dinner",
            "Document intelligence and processing"
        ]
        
        # Test batch embedding
        batch_embeddings = engine.embed_batch(test_texts)
        logger.info(f"✅ Batch processing pre-warmed with {len(batch_embeddings)} embeddings")
        
        total_time = time.time() - start_time
        logger.info(f"🎉 Model pre-warming completed successfully in {total_time:.1f}s")
        logger.info("💡 Subsequent collection processing should take 20-30s each!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model pre-warming failed: {e}")
        return False

if __name__ == "__main__":
    success = pre_warm_models()
    sys.exit(0 if success else 1)
