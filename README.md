# Assignment 4: Advanced Image Generation

APAN 5560 Applied Generative AI

This repository contains a FastAPI application with multiple generative AI models including Diffusion Models and Energy-Based Models (EBMs) for image generation on CIFAR-10 dataset, built as part of Assignment 4.

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

# Run the API
uv run fastapi dev app/main.py
```

### Option 2: Run with Docker

```bash
cd hello_world_genai
docker build -t hello_world_genai .
docker run -p 8000:8000 hello_world_genai
```

**Access the API:**
- Swagger UI: http://127.0.0.1:8000/docs
- API Root: http://127.0.0.1:8000

---

## Project Structure

```
hello_world_genai/
│
├── README.md                           # This file
├── pyproject.toml                      # Dependencies (v0.4.1)
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
│   └── ebm_model.py                    # Energy-Based Model generator (Assignment 4)
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
│   └── test_ebm_generate.py            # Quick test for EBM generation
│
├── models/                             # Trained model weights
│   ├── assignment_cnn.pth              # Trained CNN model
│   ├── mnist_gan.pth                   # Full MNIST GAN model
│   ├── mnist_gan_generator.pth         # MNIST GAN generator only
│   ├── diffusion_checkpoints/          # Diffusion training checkpoints
│   │   └── diffusion_best.pth          # Best Diffusion model (Assignment 4)
│   ├── ebm_cifar10_checkpoints/        # EBM training checkpoints
│   └── ebm_cifar10_best.pth            # EBM for CIFAR-10 (Assignment 4)
│
└── data/                               # Dataset storage (gitignored)
    ├── .gitkeep                        # Keep directory in git
    ├── MNIST/                          # MNIST dataset (auto-downloaded)
    └── cifar-10-batches-py/            # CIFAR-10 dataset (auto-downloaded)
```

---

## Assignment 4: Diffusion & Energy-Based Models

### Part 1: Diffusion Model (trained by CIFAR-10 dataset)

A UNet-based Diffusion Model that learns to generate images by reversing a noise diffusion process.

#### Architecture
- **Input:** 64×64 RGB images (resized from CIFAR-10 32×32)
- **UNet Backbone:**
  - Encoder: DownBlocks with residual connections
  - Bottleneck: ResidualBlocks
  - Decoder: UpBlocks with skip connections
- **Sinusoidal Embedding:** Encodes noise variance at each timestep
- **Diffusion Schedule:** Offset Cosine schedule for better sample quality
- **EMA:** Exponential Moving Average for stable sampling

#### Key Components
```
DiffusionModel
├── UNet (neural network backbone)
│   ├── SinusoidalEmbedding
│   ├── DownBlocks (encoder)
│   ├── ResidualBlocks (bottleneck)
│   └── UpBlocks (decoder)
├── Diffusion Schedule (noise/signal rates)
├── EMA UNet (for inference)
└── Normalizer (input normalization)
```

### Part 2: Energy-Based Model (trained by CIFAR-10 dataset)

An Energy-Based Model that learns an energy function mapping images to scalar values, using Langevin dynamics for sampling.

#### Architecture
- **Input:** 32×32 RGB images (CIFAR-10)
- **CNN Energy Function:**
  - Conv2d: in_channels → 16, k=5, s=2, p=2 + Swish
  - Conv2d: 16 → 32, k=3, s=2, p=1 + Swish
  - Conv2d: 32 → 64, k=3, s=2, p=1 + Swish
  - Conv2d: 64 → 64, k=3, s=2, p=1 + Swish
  - Flatten + Linear: 256 → 64 → 1
- **Output:** Scalar energy value

#### Training Method
- **Loss:** Contrastive Divergence (minimize real energy, maximize fake energy)
- **Sampling:** Langevin Dynamics with gradient descent
- **Buffer:** Sample buffer for efficient training

---

## API Endpoints

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


### Other Endpoints

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

## Model Training workflow for reference

### Train Diffusion Model (CIFAR-10)

```bash
cd hello_world_genai
source .venv/bin/activate  # or: uv run python ...
python scripts/train_diffusion_cifar10.py
```

**Configuration:**
- Image size: 64×64 (resized from 32×32)
- Batch size: 64
- Epochs: 50
- Learning rate: 1e-4
- Loss: L1 (MAE)

**Output:** `models/diffusion_best.pth`

### Train EBM (CIFAR-10)

```bash
cd hello_world_genai
source .venv/bin/activate
python scripts/train_ebm_cifar10.py
```

**Configuration:**
- Image size: 32×32 RGB
- Batch size: 64
- Epochs: 30
- Learning rate: 1e-4
- Langevin steps: 60
- Step size: 10.0

**Output:** `models/ebm_cifar10_best.pth`

### Training Notes

⚠️ **Training Time:**
- Diffusion Model: ~1-2 hours on GPU, ~4-6 hours on CPU
- EBM: ~2-4 hours on GPU (Langevin sampling is slow)

💡 **Tips:**
- Use GPU (CUDA/MPS) for faster training
- Monitor loss curves for convergence
- Adjust epochs based on available time

---

## Testing the API

### Using Swagger UI

1. Visit http://127.0.0.1:8000/docs
2. Navigate to the desired section (Diffusion, EBM, GAN)
3. Try the endpoints

#### Generate Diffusion Image
- Expand `GET /diffusion/generate/image`
- Set `diffusion_steps=100` (more = better quality)
- Click "Execute"

#### Generate EBM Image Grid (CIFAR-10)
- Expand `GET /ebm/generate/grid`
- Set `num_samples=9`, `nrow=3`, `steps=256`
- Click "Execute"

### Using curl

```bash
# Generate single Diffusion image
curl "http://127.0.0.1:8000/diffusion/generate/image?diffusion_steps=100" --output diffusion.png

# Generate EBM CIFAR-10 image
curl "http://127.0.0.1:8000/ebm/generate/image?steps=256" --output ebm_cifar10.png

# Get Diffusion model info
curl http://127.0.0.1:8000/diffusion/info

# Get EBM model info
curl http://127.0.0.1:8000/ebm/info
```


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
uv run fastapi dev app/main.py --port 8001


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

---
