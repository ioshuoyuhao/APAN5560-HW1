# Assignment 3: Image Generation - Generative Adversarial Networks

APAN 5560 Applied Generative AI

This repository contains the FastAPI application with MNIST GAN for generating handwritten digits, built as part of Assignment 3

---

## Quick Start

> **Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/) package manager

### Option 1: Run with uv

```bash
# Navigate to project directory
cd hello_world_genai

# (Optional) Recreate venv if has env path conflict
rm -rf .venv && uv venv && uv sync

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
├── pyproject.toml                      # Dependencies (v0.3.0)
├── uv.lock                             # Dependency lock file
├── Dockerfile                          # Docker configuration
├── .gitignore                          # Git ignore rules
├── .python-version                     # Python version specification
│
├── app/                                # FastAPI Application
│   ├── __init__.py
│   ├── main.py                         # FastAPI endpoints (all features)
│   ├── bigram_model.py                 # Bigram text generation
│   ├── embedding_model.py              # Word embeddings
│   ├── classifier_model.py             # CIFAR-10 image classifier
│   └── gan_model.py                    # MNIST GAN generator (Assignment 3)
│
├── helper_lib/                         # Reusable Neural Network Helper Library
│   ├── __init__.py                     # Package initialization
│   ├── data_loader.py                  # get_data_loader(), get_mnist_data_loader()
│   ├── model.py                        # get_model() - MLP, CNN, VAE, GAN, MNISTGAN
│   ├── trainer.py                      # train_model(), train_mnist_gan()
│   ├── evaluator.py                    # evaluate_model()
│   ├── generator.py                    # generate_samples(), generate_mnist_gan_samples()
│   └── utils.py                        # save_model(), load_model()
│
├── scripts/                            # Training scripts
│   ├── practice1_cnn.py                # CNN training script (Assignment 2)
│   └── train_mnist_gan.py              # MNIST GAN training script (Assignment 3)
│
├── models/                             # Trained model weights
│   ├── assignment_cnn.pth              # Trained CNN model
│   ├── mnist_gan.pth                   # Full MNIST GAN model (Assignment 3)
│   └── mnist_gan_generator.pth         # MNIST GAN generator only (Assignment 3)
│
└── data/                               # Dataset storage
    ├── MNIST/                          # MNIST dataset (auto-downloaded for Assignment 3)
    └── cifar-10-batches-py/            # CIFAR-10 dataset
```

---

## Assignment 3: Image Generation using GAN Architecture

Implements a GAN matching the assignment specification for MNIST digit generation:

### Generator Architecture
- **Input:** Noise vector of shape (BATCH_SIZE, 100)
- **FC Layer:** 100 → 7×7×128, then reshape to (128, 7, 7)
- **ConvTranspose2D:** 128 → 64, kernel 4, stride 2, padding 1 + BatchNorm + ReLU → 14×14
- **ConvTranspose2D:** 64 → 1, kernel 4, stride 2, padding 1 + Tanh → 28×28
- **Output:** 28×28 grayscale image

### Discriminator Architecture
- **Input:** Image of shape (1, 28, 28)
- **Conv2D:** 1 → 64, kernel 4, stride 2, padding 1 + LeakyReLU(0.2) → 14×14
- **Conv2D:** 64 → 128, kernel 4, stride 2, padding 1 + BatchNorm + LeakyReLU(0.2) → 7×7
- **Flatten + Linear:** 6272 → 1
- **Output:** Real/Fake probability

---

## API Endpoints

### GAN Endpoints (Assignment 3)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/gan/generate` | POST | Generate 1-64 handwritten digit images (base64) |
| `/gan/generate/grid` | GET | Generate a grid of digit images |
| `/gan/generate/image` | GET | Generate single digit image (PNG) |
| `/gan/info` | GET | Get GAN model architecture info |

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/generate` | POST | Generate text using bigram model |
| `/embedding` | POST | Get word embedding vector |
| `/similarity` | POST | Calculate text similarity |
| `/classify` | POST | Classify image (CIFAR-10) |
| `/classify/classes` | GET | Get CIFAR-10 class names |

---

## Model Training (Optional)

> **Note:** If `models/mnist_gan_generator.pth` already exists, you can skip training and run the API directly.

### Train MNIST GAN

```bash
cd hello_world_genai
uv run python scripts/train_mnist_gan.py
```

This will:
- Download MNIST dataset (if not exists)
- Train the MNISTGAN model for 20 epochs
- Save models to:
  - `./models/mnist_gan.pth` (full model)
  - `./models/mnist_gan_generator.pth` (generator only for API)

**Expected output:**
```
============================================================
Assignment 3: MNIST GAN Training
============================================================
[Epoch 1/20] [Batch 0/469] [D loss: 1.3863] [G loss: 0.6931]
...
Training Complete!
Model saved to: ./models/mnist_gan.pth
Generator saved to: ./models/mnist_gan_generator.pth
============================================================
```

---

## Testing the API

### Using Swagger UI

1. Visit http://127.0.0.1:8000/docs
2. Navigate to **GAN** section
3. Try these endpoints:

#### Generate Single Image (PNG)
- Expand `GET /gan/generate/image`
- Click "Try it out" → "Execute"
- The generated digit image will display directly

#### Generate Image Grid
- Expand `GET /gan/generate/grid`
- Set `num_samples=16`, `nrow=4`
- Click "Execute"
- Response contains base64-encoded grid image

#### Generate Multiple Images
- Expand `POST /gan/generate`
- Set request body: `{"num_samples": 10}`
- Click "Execute"
- Response contains list of base64-encoded images

### Using curl

```bash
# Generate single image (saves as PNG)
curl http://127.0.0.1:8000/gan/generate/image --output generated_digit.png

# Generate grid (returns JSON with base64)
curl "http://127.0.0.1:8000/gan/generate/grid?num_samples=16&nrow=4"

# Get GAN model info
curl http://127.0.0.1:8000/gan/info
```

**Response Example (POST /gan/generate):**
```json
{
  "num_samples": 1,
  "latent_dim": 100,
  "images": ["iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAAB..."]
}
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
