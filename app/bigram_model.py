"""
Bigram Language Model for Text Generation.

This module implements a simple bigram model based on the concepts from
Module02 we learned in class.

Bigram probability formula: P(w2|w1) = C(w1,w2) / C(w1)
where C(w1,w2) is the count of bigram (w1,w2) and C(w1) is the count of word w1.
"""

from collections import defaultdict, Counter
import random
import re


class BigramModel:
    """
    A simple bigram language model for text generation.
    
    The model learns word transition probabilities from a corpus and uses
    them to generate new text by sampling the next word based on bigram probabilities.
    """

    def __init__(self, corpus: list[str], frequency_threshold: int = None):
        """
        Initialize the bigram model with a corpus.
        
        Args:
            corpus: List of text strings to train on
            frequency_threshold: Minimum word frequency to include (None = include all)
        """
        # Combine all texts in corpus
        combined_text = " ".join(corpus)
        # Build vocabulary and bigram probabilities
        self.vocab, self.bigram_probs = self._analyze_bigrams(
            combined_text, frequency_threshold
        )

    def _simple_tokenizer(self, text: str, frequency_threshold: int = None) -> list[str]:
        """
        Simple tokenizer that splits text into words.
        
        Args:
            text: Input text to tokenize
            frequency_threshold: Filter out words appearing less than this count
            
        Returns:
            List of word tokens
        """
        # Convert to lowercase and extract words using regex
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not frequency_threshold:
            return tokens
        # Count word frequencies
        word_counts = Counter(tokens)
        # Filter tokens by frequency threshold
        filtered_tokens = [
            token for token in tokens if word_counts[token] >= frequency_threshold
        ]
        return filtered_tokens

    def _analyze_bigrams(
        self, text: str, frequency_threshold: int = None
    ) -> tuple[list[str], dict]:
        """
        Analyze text to compute bigram probabilities.
        
        Args:
            text: Input text to analyze
            frequency_threshold: Minimum word frequency threshold
            
        Returns:
            Tuple of (vocabulary list, bigram probabilities dict)
        """
        words = self._simple_tokenizer(text, frequency_threshold)
        bigrams = list(zip(words[:-1], words[1:]))  # Create bigrams

        # Count bigram and unigram frequencies
        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        # Compute bigram probabilities: P(w2|w1) = C(w1,w2) / C(w1)
        bigram_probs = defaultdict(dict)
        for (word1, word2), count in bigram_counts.items():
            bigram_probs[word1][word2] = count / unigram_counts[word1]

        return list(unigram_counts.keys()), dict(bigram_probs)

    def generate_text(self, start_word: str, num_words: int = 20) -> str:
        """
        Generate text based on bigram probabilities.
        
        Args:
            start_word: The word to start generation from
            num_words: Number of words to generate
            
        Returns:
            Generated text string
        """
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(num_words - 1):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:  # If no bigrams for current word, stop
                break

            # Choose next word based on probabilities
            next_word = random.choices(
                list(next_words.keys()), weights=next_words.values()
            )[0]
            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)

    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.vocab)

