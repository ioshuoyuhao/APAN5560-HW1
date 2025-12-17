# Assignment 5: Post-training an LLM

APAN 5560 Applied Generative AI

This repository contains a FastAPI application with multiple generative AI models including:
- **HW5 (NEW):** GPT2 fine-tuned with Reinforcement Learning (PPO) for Question-Answering
- Diffusion Models and Energy-Based Models (EBMs) for image generation
- RNN/LSTM text generation, CNN classification, GAN digit generation
- etc

---

## Quick Start

> **Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/) package manager

### Option 1: Run with uv

```bash
# Navigate to project directory
cd hello_world_genai

# (Optional) Recreate venv if has env path conflict
rm -rf .venv && uv venv

# update and install dependencies
uv sync
uv pip install en-core-web-lg@https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl

# Run the API (avoid using dev mode)
uv run fastapi run app/main.py 
```

**Access the API (uv):**
- Swagger UI: http://127.0.0.1:8000/docs
- API Root: http://127.0.0.1:8000

### Option 2: Run with Docker

```bash
cd hello_world_genai
docker build -t hello_world_genai .
docker run -p 8000:8000 hello_world_genai
```

**Access the API (Docker):**
- Swagger UI: http://localhost:8000/docs
- API Root: http://localhost:8000

---

## HuggingFace Hub Model Overview

The RL-trained GPT2 model for HW5 is available on HuggingFace Hub:

🤗 **[StevenHuo/StevenHuo-gpt2-squad-rl](https://huggingface.co/StevenHuo/StevenHuo-gpt2-squad-rl)**

Users cloning this Github repository and run the app can automatically download the model from HuggingFace Hub (no local training required for inference).

---

#### Test the API
```bash
# Run FastAPI server (if via docker)
docker run -p 8000:8000 hello_world_genai

# Test endpoint via curl
curl -X POST "http://localhost:8000/generate_with_llm_rl" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?", "context": "France is a country in Western Europe. Its capital is Paris."}'
```

### HW5 API Endpoints 

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate_with_llm_rl` | POST | Generate Q&A response with RL-trained model |
| `/llm_rl/info` | GET | Get RL model information |

---

## Project Structure

```
hello_world_genai/
│
├── README.md                           # This file
├── pyproject.toml                      # Dependencies (v0.5.0)
├── uv.lock                             # Dependency lock file
├── Dockerfile                          # Docker configuration
├── .gitignore                          # Git ignore rules
├── .python-version                     # Python version specification
│
├── app/                                # FastAPI Application
│   ├── __init__.py
│   ├── main.py                         # FastAPI endpoints (all features)
│   ├── bigram_model.py                 # Bigram text generation
│   ├── rnn_model.py                    # RNN (LSTM) text generation
│   ├── embedding_model.py              # Word embeddings
│   ├── classifier_model.py             # CIFAR-10 image classifier
│   ├── gan_model.py                    # MNIST GAN generator (Assignment 3)
│   ├── diffusion_model.py              # Diffusion image generator (Assignment 4)
│   ├── ebm_model.py                    # Energy-Based Model generator (Assignment 4)
│   ├── llm_model.py                    # GPT2 fine-tuned for Q&A (Module 9)
│   └── llm_rl_model.py                 # GPT2 + RL (PPO) for Q&A (Assignment 5) 
│
├── helper_lib/                         # Reusable Neural Network Helper Library
│   ├── __init__.py                     # Package initialization
│   ├── data_loader.py                  # get_data_loader(), get_mnist_data_loader()
│   ├── model.py                        # get_model() - CNN, VAE, GAN, Diffusion, EBM
│   ├── trainer.py                      # train_model(), train_diffusion(), train_ebm()
│   ├── evaluator.py                    # evaluate_model()
│   ├── generator.py                    # generate_diffusion_samples(), generate_ebm_samples()
│   └── utils.py                        # save_model(), load_model()
│
├── scripts/                            # Training & test scripts
│   ├── practice1_cnn.py                # CNN training script (Assignment 2)
│   ├── train_mnist_gan.py              # MNIST GAN training script (Assignment 3)
│   ├── train_diffusion_cifar10.py      # Diffusion model training (Assignment 4)
│   ├── train_ebm_cifar10.py            # EBM training script (Assignment 4)
│   ├── test_ebm_generate.py            # Quick test for EBM generation
│   ├── train_llm.py                    # GPT2 fine-tuning script (Module 9)
│   └── train_llm_rl.py                 # GPT2 + RL (PPO) training (Assignment 5) 
│
├── models/                             # Trained model weights
│   ├── assignment_cnn.pth              # Trained CNN model
│   ├── mnist_gan.pth                   # Full MNIST GAN model
│   ├── mnist_gan_generator.pth         # MNIST GAN generator only
│   ├── diffusion_checkpoints/          # Diffusion training checkpoints
│   │   └── diffusion_best.pth          # Best Diffusion model (Assignment 4)
│   ├── ebm_cifar10_checkpoints/        # EBM training checkpoints
│   ├── ebm_cifar10_best.pth            # EBM for CIFAR-10 (Assignment 4)
│   ├── llm_finetuned.pth               # Fine-tuned GPT2 (Module 9)
│   ├── llm_rl_finetuned.pth            # RL-trained GPT2 (Assignment 5) 
│   └── huggingface_cache/              # HuggingFace model cache (gitignored)
│
└── data/                               # Dataset storage (gitignored)
    ├── .gitkeep                        # Keep directory in git
    ├── MNIST/                          # MNIST dataset (auto-downloaded)
    ├── cifar-10-batches-py/            # CIFAR-10 dataset (auto-downloaded)
    └── huggingface_datasets/           # HuggingFace datasets cache (gitignored)
```

---

## Other API Endpoints from previous Assignments

### Diffusion Model Endpoints (Assignment 4)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/diffusion/generate` | POST | Generate 1-16 images using diffusion model |
| `/diffusion/generate/grid` | GET | Generate a grid of images |
| `/diffusion/generate/image` | GET | Generate single image (PNG) |
| `/diffusion/info` | GET | Get Diffusion model info |

### EBM Endpoints - CIFAR-10 (Assignment 4)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ebm/generate` | POST | Generate 1-16 RGB images using EBM |
| `/ebm/generate/grid` | GET | Generate a grid of RGB images |
| `/ebm/generate/image` | GET | Generate single RGB image (PNG) |
| `/ebm/info` | GET | Get EBM model info |

### LLM Endpoints (Module 9)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate_with_llm` | POST | Generate text using fine-tuned GPT2 |
| `/llm/info` | GET | Get LLM model information |

### Rest of previous Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/generate` | POST | Generate text using bigram model |
| `/generate_with_rnn` | POST | Generate text using RNN (LSTM) |
| `/embedding` | POST | Get word embedding vector |
| `/similarity` | POST | Calculate text similarity |
| `/classify` | POST | Classify image (CIFAR-10) |
| `/classify/classes` | GET | Get CIFAR-10 class names |
| `/gan/generate` | POST | Generate 1-64 handwritten digit images |
| `/gan/generate/grid` | GET | Generate a grid of digit images |
| `/gan/generate/image` | GET | Generate single digit image (PNG) |
| `/gan/info` | GET | Get GAN model architecture info |


---
### Fine-Tune LLM with RL (Assignment 5) workflow for reference 

```bash
cd hello_world_genai
source .venv/bin/activate  # or use: uv run python ...
python scripts/train_llm_rl.py --epochs 3 --num_samples 300
```

**Configuration:**
- Base Model: `openai-community/gpt2`
- Dataset: SQuAD (Stanford Question Answering Dataset)
- Training Method: PPO (Proximal Policy Optimization)
- Batch size: 1 (for MPS/GPU memory constraints)
- Learning rate: 1e-5
- Epochs: 3

**Output:** `models/llm_rl_finetuned.pth`

**Upload to HuggingFace Hub:**
```bash
# Login first
python -c "from huggingface_hub import login; login()"

# Upload (skips training if trained model already exists)
python scripts/train_llm_rl.py --skip_training --upload_to_hub "StevenHuo/StevenHuo-gpt2-squad-rl"
```

### Training Notes

⚠️ **Training Time:**
- LLM RL (HW5): ~30-60 minutes on MPS/GPU with 300 samples
---


## Troubleshooting for common issue when running FastAPI

### Issue: `No module named 'torch'` when running FastAPI

**Cause:** The `.venv` may have hardcoded path that mismatch to your local machine

**Solution:** Recreate the virtual environment:
```bash
cd hello_world_genai
rm -rf .venv
uv sync
uv pip install en-core-web-lg@https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
uv run fastapi dev app/main.py
```

### Issue: `Can't find model 'en_core_web_lg'`

**Solution:** Install spaCy model:
```bash
uv pip install en-core-web-lg@https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
```

### Issue: `Address already in use` on port 8000

**Solution:** Use different port or kill processes:
```bash
# if run with uv
uv run fastapi dev app/main.py --port 8001

# else if run with docker
docker run -p 8001:8000 hello-world-genai
```

---

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- FastAPI
- matplotlib >= 3.7.0
- Pillow
- tqdm
- transformers >= 4.30.0 (HW5: HuggingFace Transformers for GPT2)
- datasets >= 2.14.0 (HW5: HuggingFace Datasets for SQuAD)
- huggingface_hub >= 0.16.0 (HW5: Upload models to HuggingFace Hub)
- accelerate >= 0.21.0 (HW5: Efficient model training)

---
