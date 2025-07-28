#!/usr/bin/env python3
"""Check model cache status."""

import os
from pathlib import Path

def check_cache():
    """Check model cache status."""
    
    # Check HuggingFace cache
    hf_cache = Path.home() / '.cache' / 'huggingface'
    print(f"HuggingFace cache location: {hf_cache}")
    print(f"Cache exists: {hf_cache.exists()}")
    
    if hf_cache.exists():
        total_size = sum(f.stat().st_size for f in hf_cache.rglob("*") if f.is_file())
        print(f"Cache size: {total_size // (1024*1024)} MB")
        
        # List model directories
        hub_dir = hf_cache / 'hub'
        if hub_dir.exists():
            model_dirs = [d.name for d in hub_dir.iterdir() if d.is_dir()]
            print(f"Cached models: {len(model_dirs)}")
            for model_dir in model_dirs[:5]:  # Show first 5
                print(f"  - {model_dir}")
    
    # Check if our primary model is cached
    primary_model_cache = hf_cache / 'hub' / 'models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2'
    print(f"Primary model cached: {primary_model_cache.exists()}")
    
    return True

if __name__ == "__main__":
    check_cache()
