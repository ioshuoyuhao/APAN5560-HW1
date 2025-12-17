"""
LLM (GPT2) Language Model with Reinforcement Learning Fine-tuning. (Assignment 5)

This module implements a GPT2 model fine-tuned using Reinforcement Learning (PPO)
for question-answering with a specific response format.

HW5 Assignment: Post-Training an LLM using RL approach from Module 10 & 11.

The model is trained to:
1. Answer questions from the SQuAD dataset
2. Start responses with "That is a great question! "
3. End responses with " Let me know if you have any other questions."

Reference:
- HuggingFace Transformers: https://huggingface.co/docs/transformers/en/index
- SQuAD Dataset: https://huggingface.co/datasets/rajpurkar/squad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Set custom cache directories BEFORE importing transformers
def _get_project_root() -> Path:
    """Get the project root directory (hello_world_genai/)."""
    return Path(__file__).parent.parent.absolute()

PROJECT_ROOT = _get_project_root()
HF_CACHE_DIR = PROJECT_ROOT / "models" / "huggingface_cache"
DATASETS_CACHE_DIR = PROJECT_ROOT / "data" / "huggingface_datasets"

HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(DATASETS_CACHE_DIR)

# HuggingFace imports
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============== Response Format Templates ==============
RESPONSE_PREFIX = "That is a great question! "
RESPONSE_SUFFIX = " Let me know if you have any other questions."


class SQuADDataset(Dataset):
    """
    Dataset for SQuAD Q&A fine-tuning with formatted responses.
    """
    
    def __init__(self, data: List[dict], tokenizer, max_length: int = 256):
        """
        Initialize the SQuAD dataset.
        
        Args:
            data: List of dictionaries with 'question', 'context', 'answers' keys
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
        
        question = item.get('question', '')
        context = item.get('context', '')
        answers = item.get('answers', {})
        
        # Get the first answer
        if isinstance(answers, dict) and 'text' in answers:
            answer_text = answers['text'][0] if answers['text'] else ''
        elif isinstance(answers, list) and len(answers) > 0:
            answer_text = answers[0].get('text', '') if isinstance(answers[0], dict) else str(answers[0])
        else:
            answer_text = ''
        
        # Format with response template (HW5 requirement)
        formatted_answer = f"{RESPONSE_PREFIX}{answer_text}{RESPONSE_SUFFIX}"
        
        # Create training prompt
        prompt = f"Question: {question}\nContext: {context}\nAnswer: {formatted_answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'question': question,
            'answer': answer_text
        }


class RewardModel:
    """
    Reward model for RL-based fine-tuning.
    
    Implements shaped rewards based on Module 10 & 11 concepts:
    - Rewards for correct format (prefix/suffix)
    - Rewards for answer quality
    - Penalties for format violations
    """
    
    def __init__(self, prefix: str = RESPONSE_PREFIX, suffix: str = RESPONSE_SUFFIX):
        self.prefix = prefix
        self.suffix = suffix
    
    def compute_reward(self, generated_text: str, reference_answer: str = None) -> float:
        """
        Compute shaped reward for generated text.
        
        Args:
            generated_text: The model's generated response
            reference_answer: Optional reference answer for comparison
            
        Returns:
            Reward value (float)
        """
        reward = 0.0
        
        # Extract answer part after "Answer:"
        if "Answer:" in generated_text:
            answer_part = generated_text.split("Answer:")[-1].strip()
        else:
            answer_part = generated_text.strip()
        
        # Reward for starting with correct prefix
        if answer_part.startswith(self.prefix):
            reward += 5.0
        else:
            reward -= 3.0
        
        # Reward for ending with correct suffix
        if answer_part.endswith(self.suffix):
            reward += 5.0
        else:
            reward -= 3.0
        
        # Reward for containing actual content (not just template)
        content = answer_part.replace(self.prefix, "").replace(self.suffix, "").strip()
        if len(content) > 10:
            reward += 3.0
        elif len(content) > 0:
            reward += 1.0
        else:
            reward -= 2.0
        
        # Bonus for matching reference answer (if provided)
        if reference_answer and reference_answer.lower() in answer_part.lower():
            reward += 5.0
        
        return reward
    
    def compute_batch_rewards(self, generated_texts: List[str], 
                              reference_answers: List[str] = None) -> torch.Tensor:
        """Compute rewards for a batch of generated texts."""
        if reference_answers is None:
            reference_answers = [None] * len(generated_texts)
        
        rewards = [
            self.compute_reward(text, ref) 
            for text, ref in zip(generated_texts, reference_answers)
        ]
        return torch.tensor(rewards, dtype=torch.float32)


class PPOTrainer:
    """
    PPO (Proximal Policy Optimization) Trainer for LLM fine-tuning.
    
    Based on the RL concepts from Module 10 (policy gradient) and 
    Module 11 (PPO for language models).
    
    Key insight: We must separate generation (no grad) from log_prob computation (with grad).
    1. Generate tokens without gradient tracking
    2. Recompute log probabilities WITH gradients using a forward pass
    3. Use recomputed log probs for backward pass
    """
    
    def __init__(self, model, tokenizer, reward_model: RewardModel,
                 learning_rate: float = 1e-5,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        """
        Initialize PPO trainer.
        
        Args:
            model: The language model to train
            tokenizer: The tokenizer
            reward_model: Reward model for computing rewards
            learning_rate: Learning rate
            clip_epsilon: PPO clipping epsilon
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.device = next(model.parameters()).device
    
    def generate_response(self, prompt: str, max_new_tokens: int = 30) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Generate a response and return the generated token IDs.
        
        Note: Generation is done without gradients. Log probabilities are computed
        separately in a forward pass to enable gradient computation.
        """
        # Reduced max_length from 256 to 128 for memory optimization
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids)).to(self.device)
        
        # Generate without gradients (sampling process)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=False  # We'll recompute with gradients
            )
        
        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text, input_ids, generated_ids
    
    def compute_log_probs_with_grad(self, input_ids: torch.Tensor, 
                                     generated_ids: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Compute log probabilities of generated tokens WITH gradient tracking.
        
        This is the key fix: we do a forward pass through the model to get
        log probabilities that have gradient computation enabled.
        
        Returns:
            Tuple of (log_prob tensor, has_valid_grad boolean)
        """
        # Get the full sequence (input + generated)
        full_sequence = generated_ids.unsqueeze(0) if generated_ids.dim() == 1 else generated_ids
        
        # Create attention mask
        attention_mask = torch.ones_like(full_sequence)
        
        # Forward pass WITH gradients
        outputs = self.model(
            input_ids=full_sequence,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get logits and compute log probabilities
        logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]
        
        # Shift logits and labels for causal LM (predict next token)
        shift_logits = logits[:, :-1, :]  # All except last
        shift_labels = full_sequence[:, 1:]  # All except first
        
        # Compute log probabilities for the generated tokens only
        input_len = input_ids.shape[1]
        
        # Only compute log probs for generated tokens (after input)
        if input_len < shift_labels.shape[1]:
            generated_logits = shift_logits[:, input_len-1:, :]  # Logits predicting generated tokens
            generated_labels = shift_labels[:, input_len-1:]  # The generated tokens
            
            # Compute log softmax
            log_probs = F.log_softmax(generated_logits, dim=-1)
            
            # Gather log probs for actual tokens
            token_log_probs = log_probs.gather(
                dim=-1, 
                index=generated_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Sum log probs (product of probabilities in log space)
            total_log_prob = token_log_probs.sum()
            return total_log_prob, True
        else:
            # No generated tokens - return zero with valid gradient via logits
            # Use first logit to maintain gradient connection
            dummy_log_prob = logits[:, 0, 0] * 0.0
            return dummy_log_prob.sum(), False
    
    def train_step(self, prompts: List[str], reference_answers: List[str] = None) -> Dict[str, float]:
        """
        Perform one PPO training step.
        
        Args:
            prompts: List of input prompts
            reference_answers: Optional list of reference answers
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        generated_texts = []
        all_input_ids = []
        all_generated_ids = []
        
        # Step 1: Generate responses (without gradients)
        for prompt in prompts:
            text, input_ids, generated_ids = self.generate_response(prompt)
            generated_texts.append(text)
            all_input_ids.append(input_ids)
            all_generated_ids.append(generated_ids)
        
        # Compute rewards (no gradients needed)
        rewards = self.reward_model.compute_batch_rewards(generated_texts, reference_answers)
        rewards = rewards.to(self.device)
        
        # Compute advantages (normalized rewards)
        if rewards.std() > 0:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = rewards - rewards.mean()
        
        # Step 2: Recompute log probabilities WITH gradients and compute loss
        self.optimizer.zero_grad()
        
        # Collect losses in a list to avoid leaf tensor issues
        policy_losses = []
        entropy_values = []
        valid_count = 0
        
        for i, (input_ids, generated_ids) in enumerate(zip(all_input_ids, all_generated_ids)):
            # Compute log prob with gradient (returns tuple now)
            log_prob, has_valid_grad = self.compute_log_probs_with_grad(input_ids, generated_ids)
            
            if has_valid_grad:
                # Policy gradient loss: -log_prob * advantage
                policy_loss = -log_prob * advantages[i]
                policy_losses.append(policy_loss)
                
                # Track entropy (for logging)
                entropy_values.append(-log_prob.detach())
                valid_count += 1
        
        # Handle case where we have valid losses
        if policy_losses:
            # Stack and compute mean - this maintains gradient flow
            stacked_losses = torch.stack(policy_losses)
            avg_policy_loss = stacked_losses.mean()
            avg_entropy = torch.stack(entropy_values).mean() if entropy_values else torch.tensor(0.0, device=self.device)
            
            # Total loss (with entropy bonus for exploration)
            total_loss = avg_policy_loss - self.entropy_coef * avg_entropy.detach()
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Extract values before clearing tensors
            policy_loss_val = avg_policy_loss.item()
            entropy_val = avg_entropy.item()
            total_loss_val = total_loss.item()
            mean_reward_val = rewards.mean().item()
            
            # Clear intermediate tensors to free memory
            del stacked_losses, avg_policy_loss, avg_entropy, total_loss, policy_losses, entropy_values
            del generated_texts, all_input_ids, all_generated_ids, rewards, advantages
            
            # Aggressively clear GPU cache
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'policy_loss': policy_loss_val,
                'entropy': entropy_val,
                'mean_reward': mean_reward_val,
                'total_loss': total_loss_val,
                'valid_samples': valid_count
            }
        else:
            # No valid gradients - skip this step
            mean_reward_val = rewards.mean().item()
            
            # Clear memory even when skipping
            del generated_texts, all_input_ids, all_generated_ids, rewards, advantages
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print("Warning: No valid gradients in this batch, skipping update")
            return {
                'policy_loss': 0.0,
                'entropy': 0.0,
                'mean_reward': mean_reward_val,
                'total_loss': 0.0,
                'valid_samples': 0
            }


class LLMRLGenerator:
    """
    Wrapper class for RL-trained GPT2 text generation.
    
    Implements HW5 requirements:
    1. Fine-tune GPT2 using RL (PPO)
    2. Use SQuAD dataset
    3. Response format with prefix/suffix
    4. Upload to HuggingFace Hub
    """
    
    # Default HuggingFace Hub repo for the RL-trained model (HW5)
    DEFAULT_HF_REPO = "StevenHuo/StevenHuo-gpt2-squad-rl"
    
    def __init__(self, model_name: str = "openai-community/gpt2",
                 model_path: Optional[str] = None,
                 max_length: int = 256,
                 hf_repo: Optional[str] = None):
        """
        Initialize the RL-trained LLM generator.
        
        Args:
            model_name: HuggingFace model identifier for base model
            model_path: Path to load fine-tuned model weights (local)
            max_length: Maximum sequence length
            hf_repo: HuggingFace Hub repo to load fine-tuned model from
                     (e.g., 'StevenHuo/StevenHuo-gpt2-squad-rl')
                     Public repos don't require authentication token.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.hf_repo = hf_repo or self.DEFAULT_HF_REPO
        self.device = self._get_device()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(HF_CACHE_DIR)
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = None
        self.is_loaded = False
        self.is_from_hub = False  # Track if model was loaded from HF Hub
        self.reward_model = RewardModel()
        
        # Priority: local model_path > HuggingFace Hub > pretrained base model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif self.hf_repo:
            self._try_load_from_hub()
        else:
            self._load_pretrained_model()
    
    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def _load_pretrained_model(self):
        """Load the pretrained GPT2 model."""
        print(f"Loading pretrained model: {self.model_name}")
        print(f"Model cache directory: {HF_CACHE_DIR}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(HF_CACHE_DIR)
        )
        
        # Enable gradient checkpointing to reduce memory usage during backprop
        # This trades compute for memory - recomputes activations during backward pass
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for memory optimization")
        
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        print(f"Model loaded on device: {self.device}")
    
    def _try_load_from_hub(self):
        """
        Try to load a fine-tuned model from HuggingFace Hub.
        
        For public repos, no authentication token is needed.
        Falls back to pretrained base model if Hub loading fails.
        """
        try:
            print(f"Attempting to load fine-tuned model from HuggingFace Hub: {self.hf_repo}")
            print(f"Model cache directory: {HF_CACHE_DIR}")
            
            # Try loading the model from HuggingFace Hub
            # Public repos don't require token parameter
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_repo,
                cache_dir=str(HF_CACHE_DIR)
            )
            
            # Also try loading the tokenizer from the Hub (in case it was customized)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.hf_repo,
                    cache_dir=str(HF_CACHE_DIR)
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception:
                # Keep original tokenizer if Hub tokenizer fails
                pass
            
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            self.is_from_hub = True
            print(f"✅ Successfully loaded fine-tuned model from HuggingFace Hub!")
            print(f"   Repo: {self.hf_repo}")
            print(f"   Device: {self.device}")
            
        except Exception as e:
            print(f"⚠️  Could not load from HuggingFace Hub: {e}")
            print(f"   Falling back to pretrained base model: {self.model_name}")
            self._load_pretrained_model()
    
    def load_squad_dataset(self, num_samples: int = 1000,
                           split: str = "train") -> List[dict]:
        """
        Load the SQuAD dataset from HuggingFace.
        
        Dataset: https://huggingface.co/datasets/rajpurkar/squad
        
        Args:
            num_samples: Number of samples to load
            split: Dataset split to load
            
        Returns:
            List of Q&A dictionaries
        """
        try:
            from datasets import load_dataset
            
            print(f"Loading SQuAD dataset (first {num_samples} samples)...")
            print(f"Dataset cache directory: {DATASETS_CACHE_DIR}")
            
            dataset = load_dataset(
                "rajpurkar/squad",
                split=split,
                cache_dir=str(DATASETS_CACHE_DIR)
            )
            
            # Take only num_samples
            data = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                data.append({
                    'question': item['question'],
                    'context': item['context'],
                    'answers': item['answers']
                })
            
            print(f"Loaded {len(data)} samples from SQuAD dataset")
            return data
            
        except Exception as e:
            print(f"Error loading SQuAD dataset: {e}")
            return self._create_sample_qa_data()
    
    def _create_sample_qa_data(self) -> List[dict]:
        """Create sample Q&A data as fallback."""
        sample_data = [
            {
                "question": "What is machine learning?",
                "context": "Machine learning is a subset of AI that enables systems to learn.",
                "answers": {"text": ["a subset of artificial intelligence"]}
            },
            {
                "question": "What is Python?",
                "context": "Python is a high-level programming language.",
                "answers": {"text": ["a high-level programming language"]}
            },
        ]
        return sample_data * 100
    
    def create_dataloaders(self, data: List[dict],
                          batch_size: int = 4,
                          train_split: float = 0.9) -> Tuple[DataLoader, DataLoader]:
        """Create train and test DataLoaders."""
        split_idx = int(len(data) * train_split)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        train_dataset = SQuADDataset(train_data, self.tokenizer, self.max_length)
        test_dataset = SQuADDataset(test_data, self.tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Created DataLoaders: Train={len(train_dataset)}, Test={len(test_dataset)}")
        return train_loader, test_loader
    
    def fine_tune_with_rl(self, data: List[dict],
                         epochs: int = 3,
                         batch_size: int = 4,
                         learning_rate: float = 1e-5,
                         save_path: Optional[str] = None) -> List[Dict[str, float]]:
        """
        Fine-tune the model using RL (PPO).
        
        This implements the RL approach from Module 10 & 11:
        - Policy gradient updates
        - Shaped rewards for response format
        - PPO clipping for stability
        
        Args:
            data: Training data (SQuAD format)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            save_path: Path to save the model
            
        Returns:
            List of training metrics per step
        """
        if self.model is None:
            self._load_pretrained_model()
        
        # Initialize PPO trainer
        ppo_trainer = PPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            reward_model=self.reward_model,
            learning_rate=learning_rate
        )
        
        all_metrics = []
        total_batches = (len(data) + batch_size - 1) // batch_size
        print(f"Starting RL fine-tuning for {epochs} epochs...")
        print(f"Response format: '{RESPONSE_PREFIX}...[answer]...{RESPONSE_SUFFIX}'")
        print(f"Total batches per epoch: {total_batches}")
        
        for epoch in range(epochs):
            epoch_rewards = []
            epoch_losses = []
            
            # Create tqdm progress bar for batches within each epoch
            pbar = tqdm(
                range(0, len(data), batch_size),
                desc=f"Epoch {epoch + 1}/{epochs}",
                total=total_batches,
                leave=True,
                ncols=100
            )
            
            for i in pbar:
                batch = data[i:i+batch_size]
                
                # Create prompts
                prompts = [
                    f"Question: {item['question']}\nContext: {item['context']}\nAnswer:"
                    for item in batch
                ]
                
                # Get reference answers
                ref_answers = []
                for item in batch:
                    answers = item.get('answers', {})
                    if isinstance(answers, dict) and 'text' in answers:
                        ref_answers.append(answers['text'][0] if answers['text'] else '')
                    else:
                        ref_answers.append('')
                
                # PPO training step
                metrics = ppo_trainer.train_step(prompts, ref_answers)
                all_metrics.append(metrics)
                epoch_rewards.append(metrics['mean_reward'])
                epoch_losses.append(metrics['policy_loss'])
                
                # Update progress bar with live metrics
                pbar.set_postfix({
                    'reward': f"{metrics['mean_reward']:.2f}",
                    'loss': f"{metrics['policy_loss']:.4f}",
                    'valid': metrics.get('valid_samples', 'N/A')
                })
                
                # Clear GPU memory cache to prevent OOM on MPS/CUDA
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_reward = np.mean(epoch_rewards)
            avg_loss = np.mean(epoch_losses)
            print(f"✅ Epoch {epoch + 1}/{epochs} completed - Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}")
        
        self.model.eval()
        
        if save_path:
            self.save_model(save_path)
        
        return all_metrics
    
    def generate_text(self, question: str, context: str = "",
                      max_new_tokens: int = 100,
                      temperature: float = 0.8) -> str:
        """
        Generate text using the RL-trained model.
        
        Args:
            question: The question to answer
            context: Optional context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response with proper format
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model must be loaded before generating text")
        
        self.model.eval()
        
        prompt = f"Question: {question}\nContext: {context}\nAnswer: {RESPONSE_PREFIX}"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length
        )
        input_ids = inputs['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Ensure response ends with suffix if not already
        if "Answer:" in generated_text:
            answer_part = generated_text.split("Answer:")[-1].strip()
        else:
            answer_part = generated_text
        
        if not answer_part.endswith(RESPONSE_SUFFIX):
            # Try to add suffix if there's room
            answer_part = answer_part.rstrip('.') + RESPONSE_SUFFIX
        
        return f"Question: {question}\nAnswer: {answer_part}"
    
    def save_model(self, model_path: str = "models/llm_rl_finetuned.pth"):
        """Save the RL-trained model."""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
        }
        torch.save(checkpoint, model_path)
        print(f"RL-trained LLM model saved to {model_path}")
    
    def save_for_hub(self, local_dir: str = "models/gpt2-squad-rl"):
        """
        Save model in HuggingFace format for uploading to Hub.
        
        Args:
            local_dir: Local directory to save the model
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        save_path = PROJECT_ROOT / local_dir
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving model in HuggingFace format to {save_path}...")
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        print(f"Model saved to {save_path}")
        print("Ready to upload to HuggingFace Hub!")
    
    def upload_to_hub(self, repo_name: str, 
                      commit_message: str = "Upload RL fine-tuned GPT2 for Q&A"):
        """
        Upload the model to HuggingFace Hub.
        
        Args:
            repo_name: HuggingFace repo name (e.g., 'username/gpt2-squad-rl')
            commit_message: Commit message for the upload
        """
        from huggingface_hub import HfApi, get_token
        
        # Get HuggingFace token - try multiple methods
        token = get_token()
        
        if token is None:
            # Token not found, provide helpful instructions
            raise RuntimeError(
                "HuggingFace token not found!\n\n"
                "Please login using ONE of these methods:\n"
                "  Method 1: Run 'huggingface-cli login' and paste your Write token\n"
                "  Method 2: Run 'python -c \"from huggingface_hub import login; login()\"'\n"
                "  Method 3: Set environment variable HF_TOKEN=your_token\n\n"
                "Get your token at: https://huggingface.co/settings/tokens\n"
                "Make sure to create a token with WRITE access!"
            )
        
        print(f"✅ HuggingFace token found")
        
        # First save in HF format
        local_dir = PROJECT_ROOT / "models" / "hf_upload_rl"
        local_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving model to {local_dir}...")
        
        # Move model to CPU before saving to avoid MPS issues
        self.model.to('cpu')
        self.model.save_pretrained(str(local_dir))
        self.tokenizer.save_pretrained(str(local_dir))
        # Move back to original device
        self.model.to(self.device)
        
        # Upload to Hub with explicit token
        print(f"Uploading to HuggingFace Hub: {repo_name}...")
        api = HfApi(token=token)
        
        api.create_repo(repo_id=repo_name, exist_ok=True)
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_name,
            commit_message=commit_message
        )
        print(f"✅ Model uploaded to: https://huggingface.co/{repo_name}")
    
    def load_model(self, model_path: str) -> bool:
        """Load a fine-tuned model."""
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
        
        try:
            print(f"Loading RL-trained model from {model_path}")
            
            # IMPORTANT: Load checkpoint to CPU first to avoid MPS alignment issues
            # This prevents the "Unaligned blit request" error on Apple Silicon
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            self.model_name = checkpoint.get('model_name', self.model_name)
            self.max_length = checkpoint.get('max_length', self.max_length)
            
            # Load model structure on CPU first
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=str(HF_CACHE_DIR)
            )
            
            # Load state dict while model is on CPU
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Now move to target device (MPS/CUDA/CPU)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            
            print(f"RL-trained LLM model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            self._load_pretrained_model()
            return False
    
    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            "model": "GPT2 (RL Fine-tuned for Q&A - HW5)",
            "base_model": self.model_name,
            "device": str(self.device),
            "max_length": self.max_length,
            "is_loaded": self.is_loaded,
            "loaded_from_hub": self.is_from_hub,
            "huggingface_repo": self.hf_repo,
            "training_method": "PPO (Proximal Policy Optimization)",
            "response_format": {
                "prefix": RESPONSE_PREFIX,
                "suffix": RESPONSE_SUFFIX
            },
            "fine_tuning_dataset": "rajpurkar/squad (SQuAD)",
            "huggingface_model_url": f"https://huggingface.co/{self.hf_repo}",
            "huggingface_docs": "https://huggingface.co/docs/transformers/en/index",
            "cache_directories": {
                "model_cache": str(HF_CACHE_DIR),
                "dataset_cache": str(DATASETS_CACHE_DIR),
            },
            "note": "Public HF repos don't require authentication token for downloading"
        }

