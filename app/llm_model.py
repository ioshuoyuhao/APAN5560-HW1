"""
LLM (GPT2) Language Model for Question-Answering Text Generation.

This module implements a fine-tuned GPT2 model for question-answering text generation,
using HuggingFace Transformers library.

The model is fine-tuned on the Nectar dataset from HuggingFace for Q&A tasks.
Reference: https://huggingface.co/docs/transformers/en/index
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from typing import Optional, List, Tuple
from pathlib import Path

# Set custom cache directories BEFORE importing transformers
# This ensures all HuggingFace models/datasets are stored within the codebase
def _get_project_root() -> Path:
    """Get the project root directory (hello_world_genai/)."""
    # This file is at: hello_world_genai/app/llm_model.py
    # Project root is: hello_world_genai/
    return Path(__file__).parent.parent.absolute()

# Define cache directories within the project
PROJECT_ROOT = _get_project_root()
HF_CACHE_DIR = PROJECT_ROOT / "models" / "huggingface_cache"
DATASETS_CACHE_DIR = PROJECT_ROOT / "data" / "huggingface_datasets"

# Create directories if they don't exist
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set environment variables for HuggingFace cache (must be set before importing transformers)
# Note: HF_HOME is the main cache dir; TRANSFORMERS_CACHE is deprecated in v5+
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(DATASETS_CACHE_DIR)

# HuggingFace imports (after setting cache directories)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class QADataset(Dataset):
    """
    Dataset for Q&A fine-tuning.
    
    Formats question-answer pairs into a prompt format suitable for GPT2.
    """
    
    def __init__(self, data: List[dict], tokenizer, max_length: int = 128):
        """
        Initialize the Q&A dataset for fine-tuning GPT-2 model.
        
        Args:
            data: List of dictionaries with 'prompt' and 'answers' keys
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as "Question: ... Answer: ..." for Q&A fine-tuning
        prompt = item.get('prompt', '')
        # Get the first (best) answer if available
        answers = item.get('answers', [])
        if answers and len(answers) > 0:
            # Nectar dataset structure: answers is a list with 'answer' key
            if isinstance(answers[0], dict):
                answer = answers[0].get('answer', '')
            else:
                answer = str(answers[0])
        else:
            answer = ''
        
        # Create formatted text for training
        formatted_text = f"Question: {prompt}\nAnswer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # For causal LM, labels are the same as input_ids (shifted internally by the model)
        labels = input_ids.clone()
        # Mask padding tokens in labels (-100 is ignored by CrossEntropyLoss)
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class LLMTextGenerator:
    """
    Wrapper class for GPT2-based text generation with fine-tuning capability.
    
    Uses HuggingFace Transformers to load pretrained GPT2 and fine-tune
    on question-answering datasets.
    """
    
    def __init__(self, model_name: str = "openai-community/gpt2", 
                 model_path: Optional[str] = None,
                 max_length: int = 128):
        """
        Initialize the LLM text generator.
        
        Args:
            model_name: HuggingFace model identifier (default: openai-community/gpt2)
            model_path: Path to load fine-tuned model weights
            max_length: Maximum sequence length for generation
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = self._get_device()
        
        # Initialize tokenizer (cached in project directory)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(HF_CACHE_DIR)
        )
        
        # GPT2 doesn't have a pad token by default, use EOS token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = None
        self.is_loaded = False
        
        # Try to load fine-tuned model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Load base pretrained model
            self._load_pretrained_model()
    
    def _get_device(self) -> torch.device:
        """Get the best available device (MPS/CUDA/CPU)."""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def _load_pretrained_model(self):
        """Load the pretrained GPT2 model from HuggingFace.
        
        Model is cached in: models/huggingface_cache/ within the project.
        """
        print(f"Loading pretrained model: {self.model_name}")
        print(f"Model cache directory: {HF_CACHE_DIR}")
        
        # Explicitly set cache_dir to ensure model is saved in project directory
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(HF_CACHE_DIR)
        )
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        print(f"Model loaded on device: {self.device}")
    
    def load_nectar_dataset(self, num_samples: int = 1000, 
                           split: str = "train") -> List[dict]:
        """
        Load the Nectar dataset from HuggingFace.
        
        Dataset is cached in: data/huggingface_datasets/ within the project.
        
        Args:
            num_samples: Number of samples to load (for faster training)
            split: Dataset split to load
            
        Returns:
            List of Q&A dictionaries
        """
        try:
            from datasets import load_dataset
            
            print(f"Loading Nectar dataset (first {num_samples} samples)...")
            print(f"Dataset cache directory: {DATASETS_CACHE_DIR}")
            
            # Load streaming to avoid downloading entire dataset
            # Cache directory is set via HF_DATASETS_CACHE environment variable
            dataset = load_dataset(
                "berkeley-nest/Nectar", 
                split=split, 
                streaming=True,
                trust_remote_code=True,
                cache_dir=str(DATASETS_CACHE_DIR)  # Explicitly set cache dir
            )
            
            # Take only num_samples
            data = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                data.append(item)
            
            print(f"Loaded {len(data)} samples from Nectar dataset")
            return data
            
        except Exception as e:
            print(f"Error loading Nectar dataset: {e}")
            # Fallback: create sample Q&A data
            return self._create_sample_qa_data()
    
    def _create_sample_qa_data(self) -> List[dict]:
        """Create sample Q&A data as fallback."""
        sample_data = [
            {"prompt": "What is machine learning?", 
             "answers": [{"answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."}]},
            {"prompt": "What is Python?",
             "answers": [{"answer": "Python is a high-level, interpreted programming language known for its simplicity and readability."}]},
            {"prompt": "What is deep learning?",
             "answers": [{"answer": "Deep learning is a type of machine learning based on artificial neural networks with multiple layers."}]},
            {"prompt": "What is a transformer?",
             "answers": [{"answer": "A transformer is a neural network architecture that uses self-attention mechanisms to process sequential data."}]},
            {"prompt": "What is GPT?",
             "answers": [{"answer": "GPT (Generative Pre-trained Transformer) is a type of large language model developed by OpenAI."}]},
        ]
        return sample_data * 200  # Repeat to have more samples
    
    def create_dataloaders(self, data: List[dict], 
                          batch_size: int = 8,
                          train_split: float = 0.9) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and test DataLoaders from the Q&A data.
        
        Args:
            data: List of Q&A dictionaries
            batch_size: Batch size for training
            train_split: Fraction of data for training
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Split data
        split_idx = int(len(data) * train_split)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        # Create datasets
        train_dataset = QADataset(train_data, self.tokenizer, self.max_length)
        test_dataset = QADataset(test_data, self.tokenizer, self.max_length)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Created DataLoaders: Train={len(train_dataset)}, Test={len(test_dataset)}")
        return train_loader, test_loader
    
    def fine_tune(self, train_loader: DataLoader, 
                  epochs: int = 3,
                  learning_rate: float = 5e-5,
                  save_path: Optional[str] = None) -> List[float]:
        """
        Fine-tune the GPT2 model on Q&A data.
        
        Args:
            train_loader: Training DataLoader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            save_path: Path to save the fine-tuned model
            
        Returns:
            List of loss values per epoch
        """
        if self.model is None:
            self._load_pretrained_model()
        
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        losses = []
        print(f"Starting fine-tuning for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        
        self.model.eval()
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        return losses
    
    def generate_text(self, prompt: str, 
                      max_new_tokens: int = 50,
                      temperature: float = 1.0,
                      top_k: int = 50,
                      top_p: float = 0.95,
                      do_sample: bool = True) -> str:
        """
        Generate text using the model.
        
        Args:
            prompt: Input prompt/question
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling (vs greedy decoding)
            
        Returns:
            Generated text string
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model must be loaded before generating text")
        
        self.model.eval()
        
        # Format prompt for Q&A if it looks like a question
        if not prompt.startswith("Question:"):
            formatted_prompt = f"Question: {prompt}\nAnswer:"
        else:
            formatted_prompt = prompt
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def save_model(self, model_path: str = "models/llm_finetuned.pth"):
        """
        Save the fine-tuned model.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model state and config
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
        }
        torch.save(checkpoint, model_path)
        print(f"LLM model saved to {model_path}")
    
    def load_model(self, model_path: str = "models/llm_finetuned.pth") -> bool:
        """
        Load a fine-tuned model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
        
        try:
            print(f"Loading fine-tuned model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Restore config
            self.model_name = checkpoint.get('model_name', self.model_name)
            self.max_length = checkpoint.get('max_length', self.max_length)
            
            # Load model architecture (use project cache directory)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=str(HF_CACHE_DIR)
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            
            print(f"LLM model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load LLM model: {e}")
            # Fallback to base model
            self._load_pretrained_model()
            return False
    
    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            "model": "GPT2 (Fine-tuned for Q&A)",
            "base_model": self.model_name,
            "device": str(self.device),
            "max_length": self.max_length,
            "is_loaded": self.is_loaded,
            "vocab_size": self.tokenizer.vocab_size,
            "architecture": {
                "type": "Causal Language Model",
                "layers": "12 transformer blocks",
                "hidden_size": 768,
                "attention_heads": 12,
            },
            "fine_tuning_dataset": "berkeley-nest/Nectar (Q&A)",
            "huggingface_docs": "https://huggingface.co/docs/transformers/en/index",
            "cache_directories": {
                "model_cache": str(HF_CACHE_DIR),
                "dataset_cache": str(DATASETS_CACHE_DIR),
            }
        }

