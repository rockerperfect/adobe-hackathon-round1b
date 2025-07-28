"""
Similarity - Semantic similarity calculations for embeddings.

Purpose:
    Provides functions for cosine similarity and other relevant metrics for comparing embedding vectors.

Structure:
    - cosine_similarity: main similarity function
    - Additional metrics as needed

Dependencies:
    - None (uses only standard library)

Integration Points:
    - Used by SectionRanker, RelevanceCalculator, SubsectionExtractor, etc.

NOTE: All functions handle edge cases and are fully documented.
"""

from typing import List, Optional
import math

def cosine_similarity(vec1: Optional[List[float]], vec2: Optional[List[float]]) -> float:
    """
    Compute the cosine similarity between two embedding vectors.

    Args:
        vec1 (list): First embedding vector.
        vec2 (list): Second embedding vector.

    Returns:
        float: Cosine similarity score in [-1, 1]. Returns 0.0 if input is invalid.

    Limitations:
        - Returns 0.0 if either vector is None, empty, or mismatched in length.
        - Does not normalize input vectors.
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)

# TODO: Add additional similarity metrics (e.g., Euclidean, Manhattan) if needed.
