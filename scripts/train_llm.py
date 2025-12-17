#!/usr/bin/env python3
"""
Script to fine-tune GPT2 on the Nectar Q&A dataset.

This script fine-tunes the openai-community/gpt2 model for question-answering
using the HuggingFace Transformers library and the Nectar dataset.

Usage:
    python scripts/train_llm.py --epochs 3 --batch_size 8 --num_samples 1000

Reference: https://huggingface.co/docs/transformers/en/index
Dataset: https://huggingface.co/datasets/berkeley-nest/Nectar
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.llm_model import LLMTextGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT2 on Nectar Q&A dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model identifier (default: openai-community/gpt2)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples from Nectar dataset (default: 1000)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="models/llm_finetuned.pth",
        help="Path to save the fine-tuned model (default: models/llm_finetuned.pth)"
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default="What is machine learning?",
        help="Test prompt for generation after training"
    )
    return parser.parse_args()


def main():
    """Main function to run the fine-tuning."""
    args = parse_args()
    
    print("=" * 60)
    print("GPT2 Fine-tuning for Q&A (Module 9 Activity)")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Num samples: {args.num_samples}")
    print(f"Max length: {args.max_length}")
    print(f"Save path: {args.save_path}")
    print("=" * 60)
    
    # Step 1: Initialize the LLM generator with base model
    print("\n[Step 1] Loading pretrained GPT2 model...")
    llm_generator = LLMTextGenerator(
        model_name=args.model_name,
        max_length=args.max_length
    )
    
    # Step 2: Load and prepare the Nectar dataset
    print(f"\n[Step 2] Loading Nectar dataset ({args.num_samples} samples)...")
    data = llm_generator.load_nectar_dataset(num_samples=args.num_samples)
    
    # Step 3: Create DataLoaders
    print("\n[Step 3] Creating train and test DataLoaders...")
    train_loader, test_loader = llm_generator.create_dataloaders(
        data,
        batch_size=args.batch_size,
        train_split=0.9
    )
    
    # Step 4: Fine-tune the model
    print("\n[Step 4] Starting fine-tuning...")
    losses = llm_generator.fine_tune(
        train_loader=train_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=args.save_path
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Model saved to: {args.save_path}")
    print("=" * 60)
    
    # Step 5: Test generation
    print("\n[Step 5] Testing generation...")
    print(f"Prompt: {args.test_prompt}")
    print("-" * 40)
    
    generated = llm_generator.generate_text(
        prompt=args.test_prompt,
        max_new_tokens=50,
        temperature=0.8
    )
    print(f"Generated: {generated}")
    print("=" * 60)
    
    # Print usage instructions
    print("\n" + "=" * 60)
    print("To use the fine-tuned model in FastAPI:")
    print("1. Make sure the model is saved at: models/llm_finetuned.pth")
    print("2. Start the API: uvicorn app.main:app --reload")
    print("3. Send POST request to /generate_with_llm endpoint")
    print("=" * 60)


if __name__ == "__main__":
    main()

