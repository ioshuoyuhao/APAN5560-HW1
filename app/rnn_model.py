"""
RNN (LSTM) Language Model for Text Generation.

This module implements an LSTM-based language model for text generation,
based on the concepts from Module 6/7 RNN practical.

The model learns to predict the next word in a sequence using LSTM architecture.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
import os
from collections import Counter
import requests


class LSTMModel(nn.Module):
    """
    LSTM-based language model for text generation.
    
    Architecture:
    - Embedding layer: vocab_size -> embedding_dim
    - LSTM layer: embedding_dim -> hidden_dim
    - Fully connected layer: hidden_dim -> vocab_size
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 100, hidden_dim: int = 128):
        """
        Initialize the LSTM model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
        """
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            hidden: Optional hidden state tuple (h, c)
            
        Returns:
            output: Predictions of shape (batch_size, seq_len, vocab_size)
            hidden: Updated hidden state tuple
        """
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


class TextDataset(Dataset):
    """Dataset for text sequences used in training."""
    
    def __init__(self, data: list, seq_len: int = 30):
        """
        Initialize the dataset.
        
        Args:
            data: List of encoded token indices
            seq_len: Length of each sequence
        """
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx + self.seq_len]),
            torch.tensor(self.data[idx + 1:idx + self.seq_len + 1])
        )


class RNNTextGenerator:
    """
    Wrapper class for RNN-based text generation.
    
    Handles vocabulary construction, model training, and text generation.
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 100, 
                 hidden_dim: int = 128, seq_len: int = 30):
        """
        Initialize the RNN text generator.
        
        Args:
            vocab_size: Maximum vocabulary size
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
            seq_len: Sequence length for training
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        self.vocab = {}
        self.inv_vocab = {}
        self.model = None
        self.is_trained = False
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text by removing special characters and lowercasing."""
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.lower()
    
    def _build_vocabulary(self, tokens: list) -> None:
        """
        Build vocabulary from tokens.
        
        Args:
            tokens: List of word tokens
        """
        counter = Counter(tokens)
        # Assign indices 0 and 1 to special tokens, rest based on frequency
        self.vocab = {
            word: idx + 2 
            for idx, (word, _) in enumerate(counter.most_common(self.vocab_size - 2))
        }
        self.vocab["<PAD>"] = 0
        self.vocab["<UNK>"] = 1
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
    
    def load_and_prepare_data(self, url: str = None, text: str = None) -> list:
        """
        Load text data and prepare for training.
        
        Args:
            url: URL to download text from (default: Count of Monte Cristo)
            text: Direct text input (alternative to URL)
            
        Returns:
            List of encoded token indices
        """
        if text is None:
            # Default: Load Count of Monte Cristo from Project Gutenberg
            if url is None:
                url = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"
            response = requests.get(url)
            text = response.text
            
            # Keep only the main body (remove header/footer)
            start_idx = text.find("Chapter 1.")
            end_idx = text.rfind("Chapter 5.")
            text = text[start_idx:end_idx]
        
        # Preprocess
        text = self._preprocess_text(text)
        
        # Tokenize
        tokens = text.split()
        
        # Build vocabulary
        self._build_vocabulary(tokens)
        
        # Encode tokens
        encoded = [self.vocab.get(word, self.vocab["<UNK>"]) for word in tokens]
        
        return encoded
    
    def train(self, epochs: int = 15, batch_size: int = 64, 
              url: str = None, text: str = None) -> list:
        """
        Train the LSTM model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            url: URL to download training text
            text: Direct text input for training
            
        Returns:
            List of loss values per epoch
        """
        # Load and prepare data
        encoded = self.load_and_prepare_data(url=url, text=text)
        
        # Create dataset and dataloader
        dataset = TextDataset(encoded, self.seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        actual_vocab_size = len(self.vocab)
        self.model = LSTMModel(
            vocab_size=actual_vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # Training loop
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in dataloader:
                outputs, _ = self.model(inputs)
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)), 
                    targets.view(-1)
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        return losses
    
    def generate_text(self, seed_text: str, length: int = 50, 
                      temperature: float = 1.0) -> str:
        """
        Generate text using the trained model.
        
        Args:
            seed_text: Starting text for generation
            length: Number of words to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated text string
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before generating text")
        
        self.model.eval()
        words = seed_text.lower().split()
        input_ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in words]
        input_tensor = torch.tensor(input_ids).unsqueeze(0)
        hidden = None
        
        with torch.no_grad():
            for _ in range(length):
                output, hidden = self.model(input_tensor, hidden)
                logits = output[0, -1] / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                words.append(self.inv_vocab.get(next_id, "<UNK>"))
                
                # Extend input sequence with new token
                input_ids.append(next_id)
                input_tensor = torch.tensor(input_ids).unsqueeze(0)
        
        return " ".join(words)
    
    def get_vocab_size(self) -> int:
        """Return the actual vocabulary size."""
        return len(self.vocab)
    
    def is_model_trained(self) -> bool:
        """Check if the model has been trained."""
        return self.is_trained
    
    def save_model(self, model_path: str = "models/rnn_model.pth"):
        """
        Save the trained model and vocabulary.
        
        Args:
            model_path: Path to save the model weights
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before saving")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model state, vocabulary, and config
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'inv_vocab': self.inv_vocab,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'seq_len': self.seq_len,
        }
        torch.save(checkpoint, model_path)
        print(f"RNN model saved to {model_path}")
    
    def load_model(self, model_path: str = "models/rnn_model.pth") -> bool:
        """
        Load a pre-trained model and vocabulary.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(model_path):
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Restore vocabulary
            self.vocab = checkpoint['vocab']
            self.inv_vocab = checkpoint['inv_vocab']
            
            # Restore config
            self.vocab_size = checkpoint.get('vocab_size', self.vocab_size)
            self.embedding_dim = checkpoint.get('embedding_dim', self.embedding_dim)
            self.hidden_dim = checkpoint.get('hidden_dim', self.hidden_dim)
            self.seq_len = checkpoint.get('seq_len', self.seq_len)
            
            # Initialize and load model
            actual_vocab_size = len(self.vocab)
            self.model = LSTMModel(
                vocab_size=actual_vocab_size,
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.is_trained = True
            
            print(f"RNN model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Failed to load RNN model: {e}")
            return False

