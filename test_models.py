#!/usr/bin/env python3
"""Quick test to verify ML models are working."""

import sys

def test_models():
    """Test all ML models are working."""
    print("Testing ML models...")
    
    try:
        # Test sentence transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        test_sentences = ["Hello world", "Bonjour le monde"]
        embeddings = model.encode(test_sentences)
        print(f"✅ Sentence Transformers: {embeddings.shape}")
        
        # Test language detection
        from langdetect import detect
        result = detect("This is a test sentence in English")
        print(f"✅ Language Detection: {result}")
        
        # Test NLTK
        import nltk
        print("✅ NLTK: Available")
        
        print("\n🎉 All models are working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_models()
    sys.exit(0 if success else 1)
