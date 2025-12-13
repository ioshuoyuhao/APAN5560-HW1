"""
Word Embedding Model using spaCy.

This module implements word embedding functionality based on the concepts from
Module 2 Practice 3 - Word Embeddings.

It provides:
- Word embedding extraction (300-dimensional vectors)
- Word similarity calculation
- Sentence similarity calculation
- Vector arithmetic for semantic relationships
"""

import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingModel:
    """
    A word embedding model using spaCy's pre-trained word vectors.
    
    Uses the 'en_core_web_lg' model which contains 300-dimensional
    word vectors trained on web text.
    """

    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Initialize the embedding model by loading spaCy model.
        
        Args:
            model_name: Name of the spaCy model to load (default: en_core_web_lg)
        """
        self.nlp = spacy.load(model_name)
        self.vector_dim = 300  # en_core_web_lg uses 300-dimensional vectors

    def get_embedding(self, text: str) -> list[float]:
        """
        Get the embedding vector for a word or text.
        
        Args:
            text: Input word or text to get embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        doc = self.nlp(text)
        return doc.vector.tolist()

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two words or texts.
        
        Uses spaCy's built-in similarity method which computes
        cosine similarity between the vectors.
        
        Args:
            text1: First word or text
            text2: Second word or text
            
        Returns:
            Similarity score between 0 and 1
        """
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        return float(doc1.similarity(doc2))

    def vector_arithmetic(
        self, word1: str, word2: str, word3: str, word4: str
    ) -> float:
        """
        Perform vector arithmetic: (word1 + word2 - word3) similarity with word4.
        
        This demonstrates semantic relationships like:
        - king - man + woman ≈ queen
        - spain + paris - france ≈ madrid
        
        Args:
            word1: First word (base)
            word2: Second word (to add)
            word3: Third word (to subtract)
            word4: Fourth word (to compare result with)
            
        Returns:
            Cosine similarity between (word1 + word2 - word3) and word4
        """
        vec1 = self.nlp(word1).vector
        vec2 = self.nlp(word2).vector
        vec3 = self.nlp(word3).vector
        vec4 = self.nlp(word4).vector
        
        # Compute: word1 + word2 - word3
        result_vector = vec1 + (vec2 - vec3)
        
        # Calculate cosine similarity with word4
        similarity = cosine_similarity([result_vector], [vec4])[0][0]
        return float(similarity)

    def get_vector_dimension(self) -> int:
        """Return the dimension of the word vectors."""
        return self.vector_dim
