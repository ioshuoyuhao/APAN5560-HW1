#!/usr/bin/env python3
"""
HW5: RL-based Fine-tuning of GPT2 for Question-Answering.

This script fine-tunes the openai-community/gpt2 model using Reinforcement Learning (PPO)
on the SQuAD dataset with shaped rewards for a specific response format.

Assignment Requirements:
1. Fine-tune GPT2 using RL approach (Module 10 & 11)
2. Use SQuAD dataset: https://huggingface.co/datasets/rajpurkar/squad
3. Train model to respond with specific format:
   - Start with: "That is a great question! "
   - End with: " Let me know if you have any other questions."
4. Upload to HuggingFace Hub

Usage:
    # Train the model (skips training if trained model already exists)
    python scripts/train_llm_rl.py --epochs 3 --num_samples 500
    
    # Force re-training even if model exists
    python scripts/train_llm_rl.py --epochs 3 --force_train
    
    # Upload existing trained model to HuggingFace Hub (no training required for this time since the model is already trained)
    python scripts/train_llm_rl.py --skip_training --upload_to_hub "your-username/gpt2-squad-rl"
    
    # Re-Train the model and upload to HuggingFace Hub
    python scripts/train_llm_rl.py --epochs 3 --force_train --upload_to_hub "your-username/gpt2-squad-rl"

Reference:
- HuggingFace Transformers: https://huggingface.co/docs/transformers/en/index
- SQuAD Dataset: https://huggingface.co/datasets/rajpurkar/squad
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.llm_rl_model import LLMRLGenerator, RESPONSE_PREFIX, RESPONSE_SUFFIX


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HW5: Fine-tune GPT2 on SQuAD with RL (PPO)"
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
        default=1,
        help="Batch size for training (default: 1 for MPS memory constraints)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=300,
        help="Number of samples from SQuAD dataset (default: 300 for faster training)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128 for MPS memory constraints)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="models/llm_rl_finetuned.pth",
        help="Path to save the fine-tuned model (default: models/llm_rl_finetuned.pth)"
    )
    parser.add_argument(
        "--upload_to_hub",
        type=str,
        default=None,
        help="HuggingFace repo name to upload (e.g., 'StevenHuo/StevenHuo-gpt2-squad-rl')"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only load existing model (for upload or testing)"
    )
    parser.add_argument(
        "--force_train",
        action="store_true",
        help="Force re-training even if trained model already exists"
    )
    parser.add_argument(
        "--test_question",
        type=str,
        default="What is the capital of France?",
        help="Test question for generation after training"
    )
    parser.add_argument(
        "--test_context",
        type=str,
        default="France is a country in Western Europe. Its capital is Paris, which is known for the Eiffel Tower.",
        help="Test context for generation after training"
    )
    return parser.parse_args()


def main():
    """Main function to run RL-based fine-tuning."""
    args = parse_args()
    
    # Check if trained model already exists
    model_exists = os.path.exists(args.save_path)
    
    print("=" * 70)
    print("HW5: GPT2 Fine-tuning with Reinforcement Learning (PPO)")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Dataset: SQuAD (rajpurkar/squad)")
    print(f"Training Method: PPO (Proximal Policy Optimization)")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Num samples: {args.num_samples}")
    print(f"Max length: {args.max_length}")
    print(f"Save path: {args.save_path}")
    print(f"Trained model exists: {'✅ Yes' if model_exists else '❌ No'}")
    print(f"Skip training: {args.skip_training}")
    print(f"Force re-train: {args.force_train}")
    print("-" * 70)
    print("Response Format:")
    print(f"  Prefix: '{RESPONSE_PREFIX}'")
    print(f"  Suffix: '{RESPONSE_SUFFIX}'")
    print("=" * 70)
    
    # Determine whether to train this time (skip training if the model already exists)
    should_train = True
    if args.skip_training:
        should_train = False
        print("\n⏭️  Skipping training (--skip_training flag set)")
    elif model_exists and not args.force_train:
        should_train = False
        print(f"\n✅ Found existing trained model at: {args.save_path}")
        print("   Skipping training. Use --force_train to re-train.")
    elif args.force_train:
        print("\n🔄 Force re-training enabled (--force_train flag set)")
    
    # Step 1: Initialize the RL generator
    if model_exists and not args.force_train:
        print("\n[Step 1] Loading existing trained model...")
        llm_generator = LLMRLGenerator(
            model_name=args.model_name,
            model_path=args.save_path,
            max_length=args.max_length
        )
    else:
        print("\n[Step 1] Loading pretrained GPT2 base model...")
        llm_generator = LLMRLGenerator(
            model_name=args.model_name,
            max_length=args.max_length
        )
    
    # Step 2 & 3: Training (if needed)
    if should_train:
        # Step 2: Load and prepare the SQuAD dataset
        print(f"\n[Step 2] Loading SQuAD dataset ({args.num_samples} samples)...")
        data = llm_generator.load_squad_dataset(num_samples=args.num_samples)
        
        # Show sample data
        if data:
            print("\nSample Q&A from dataset:")
            sample = data[0]
            print(f"  Question: {sample['question'][:80]}...")
            print(f"  Context: {sample['context'][:80]}...")
            answers = sample.get('answers', {})
            if isinstance(answers, dict) and 'text' in answers:
                print(f"  Answer: {answers['text'][0][:80] if answers['text'] else 'N/A'}...")
        
        # Step 3: RL Fine-tuning with PPO
        print("\n[Step 3] Starting RL fine-tuning with PPO...")
        print("Reward shaping:")
        print("  +5: Starts with correct prefix")
        print("  +5: Ends with correct suffix")
        print("  +3: Contains meaningful content")
        print("  +5: Contains reference answer")
        print("  -3: Missing prefix or suffix")
        print("-" * 40)
        
        metrics = llm_generator.fine_tune_with_rl(
            data=data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_path=args.save_path
        )
        
        # Compute final stats
        if metrics:
            final_reward = sum(m['mean_reward'] for m in metrics[-10:]) / min(10, len(metrics))
            print(f"\n" + "=" * 70)
            print("Training Complete!")
            print(f"Final Average Reward (last 10 steps): {final_reward:.4f}")
            print(f"Model saved to: {args.save_path}")
            print("=" * 70)
    else:
        print("\n[Step 2-3] ⏭️  Training skipped (using existing model)")
    
    # Step 4: Test generation
    print("\n[Step 4] Testing generation with trained model...")
    print(f"Question: {args.test_question}")
    print(f"Context: {args.test_context[:80]}...")
    print("-" * 40)
    
    generated = llm_generator.generate_text(
        question=args.test_question,
        context=args.test_context,
        max_new_tokens=100,
        temperature=0.8
    )
    print(f"Generated Response:\n{generated}")
    print("=" * 70)
    
    # Step 5: Upload to HuggingFace Hub (if requested)
    if args.upload_to_hub:
        print(f"\n[Step 5] Uploading to HuggingFace Hub: {args.upload_to_hub}")
        print("-" * 40)
        print("Make sure you're logged in: huggingface-cli login")
        
        try:
            llm_generator.upload_to_hub(args.upload_to_hub)
            print(f"\n✅ Model uploaded successfully!")
            print(f"Access at: https://huggingface.co/{args.upload_to_hub}")
        except Exception as e:
            print(f"\n❌ Upload failed: {e}")
            print("\nTo upload manually:")
            print("1. Run: huggingface-cli login")
            print("2. Run this script again with --upload_to_hub")
    else:
        print("\n" + "=" * 70)
        print("To upload to HuggingFace Hub, run:")
        print(f"  python scripts/train_llm_rl.py --upload_to_hub 'your-username/gpt2-squad-rl'")
        print("\nOr after training:")
        print("  from app.llm_rl_model import LLMRLGenerator")
        print("  gen = LLMRLGenerator(model_path='models/llm_rl_finetuned.pth')")
        print("  gen.upload_to_hub('your-username/gpt2-squad-rl')")
        print("=" * 70)
    
    # Print usage instructions
    print("\n" + "=" * 70)
    print("To use the RL-trained model in FastAPI:")
    print("1. Make sure the model is saved at: models/llm_rl_finetuned.pth")
    print("2. Start the API: uvicorn app.main:app --reload")
    print("3. Send POST request to /generate_with_llm_rl endpoint")
    print("=" * 70)


if __name__ == "__main__":
    main()

