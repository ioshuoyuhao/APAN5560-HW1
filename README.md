# Module 4 Class Activity & Assignment 2 Practices Codebase

APAN 5560 Applied Generative AI

This repository contains the helper library for Module 4: Modern Deep Learning Architectures, and practice question implementation for HW2.

---

## Quick Start

> **Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/) package manager

```bash
# Clone and navigate to API directory
git clone <repository-url>
cd "class activity and first part Practices for HW2/hello_world_genai"

# (optional)Recreate venv if it exists to avoid path conflicts issues like `No module named 'torch'` when running FastAPI )
rm -rf .venv 2>/dev/null
# install dependencies
uv sync
uv pip install en-core-web-lg@https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl

# Run the API
uv run fastapi dev app/main.py
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
├── pyproject.toml                      # Dependencies
├── uv.lock                             # Dependency lock file
├── Dockerfile                          # Docker configuration
├── .gitignore                          # Git ignore rules
├── .python-version                     # Python version specification
│
├── app/                                # FastAPI Application
│   ├── __init__.py
│   ├── main.py                         # FastAPI endpoints
│   ├── bigram_model.py                 # Bigram text generation (HW1)
│   ├── embedding_model.py              # Word embeddings (HW1)
│   └── classifier_model.py             # CIFAR-10 image classifier (HW2)
│
├── helper_lib/                         # Reusable Neural Network Helper Library
│   ├── __init__.py                     # Package initialization
│   ├── data_loader.py                  # get_data_loader() - CIFAR-10 data loading
│   ├── model.py                        # get_model() - MLP, CNN, EnhancedCNN, AssignmentCNN
│   ├── trainer.py                      # train_model() - Training loop
│   ├── evaluator.py                    # evaluate_model() - Model evaluation
│   └── utils.py                        # save_model(), load_model() - Utilities
│
├── scripts/                            # Training and utility scripts
│   └── practice1_cnn.py                # Practice 1: CNN training script
│
├── models/                             # Trained model weights
│   └── assignment_cnn.pth              # Trained CNN model (~3.2MB)
│
└── data/                               # Dataset storage
    ├── cifar-10-python.tar.gz          # CIFAR-10 dataset archive
    └── cifar-10-batches-py/            # Extracted CIFAR-10 dataset
        ├── batches.meta
        ├── data_batch_1
        ├── data_batch_2
        ├── data_batch_3
        ├── data_batch_4
        ├── data_batch_5
        ├── test_batch
        └── readme.html
```

---

## Practice 1: CNN Architecture

Implements a CNN matching the assignment specification:

**Architecture:**
- Input: RGB image 64×64×3
- Conv2D: 16 filters, 3×3, stride 1, padding 1 → ReLU → MaxPool 2×2
- Conv2D: 32 filters, 3×3, stride 1, padding 1 → ReLU → MaxPool 2×2
- Flatten → FC(100) → ReLU → FC(10)

---

## Practice 2: Model Deployment (API)

FastAPI application with endpoints for:
- Bigram text generation
- Word embeddings & similarity
- **CIFAR-10 image classification** (HW2)

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/generate` | POST | Generate text using bigram model |
| `/embedding` | POST | Get word embedding vector |
| `/similarity` | POST | Calculate text similarity |
| `/vector-arithmetic` | POST | Perform vector arithmetic |
| `/classify` | POST | **Classify image (CIFAR-10)** |
| `/classify/classes` | GET | **Get CIFAR-10 class names** |

---

## Model Training Process for reference (IF you want to Re-train the model)

### Step 1: Train the CNN Model

> **Note:** If `models/assignment_cnn.pth` exists, u can skip to Step 2.

```bash
uv run python scripts/practice1_cnn.py
```

This will:
- Download CIFAR-10 dataset (if not exists)
- Resize images to 64×64
- Train the AssignmentCNN model for 10 epochs
- Save the model to `./models/assignment_cnn.pth`

**Expected output:**
```
Training Complete!
Final Test Accuracy: ~65%
Model saved to ./models/assignment_cnn.pth
```

### Step 2: Install API Dependencies

```bash
cd hello_world_genai
rm -rf .venv 2>/dev/null   # Remove existing venv to avoid path conflicts if needed (optional)
uv sync
uv pip install en-core-web-lg@https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
```

### Step 3: Run the API

```bash
uv run fastapi dev app/main.py
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

### Step 4: Test the API

- **Swagger UI**: http://127.0.0.1:8000/docs
- **API Root**: http://127.0.0.1:8000
- **ReDoc**: http://127.0.0.1:8000/redoc

#### Test Image Classification

1. Visit http://127.0.0.1:8000/docs
2. Expand `POST /classify`
3. Click "Try it out"
4. Upload an image file
5. Click "Execute"

**Response Example:**
```json
{
  "filename": "cat.jpg",
  "predicted_class": "cat",
  "confidence": 0.8542,
  "top5_predictions": [
    {"class": "cat", "probability": 0.8542},
    {"class": "dog", "probability": 0.0823}
  ]
}
```

**CIFAR-10 Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## Docker Deployment

```bash
cd hello_world_genai
docker build -t hello_world_genai .
docker run -p 8000:8000 hello_world_genai
```

Visit: http://0.0.0.0:8000/docs

---

## Troubleshooting

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
# or
pkill -f "uvicorn app.main"
```

### Issue: Model file not found (`assignment_cnn.pth`)

**Solution:** Train the model first:
```bash
uv run python scripts/practice1_cnn.py
```

---

## Development History

| Date | Update |
|------|--------|
| 2025-12-13 | Added CIFAR-10 image classifier endpoint (Practice 2) |
| 2025-12-13 | Created AssignmentCNN model and training script (Practice 1) |
| 2025-12-13 | Built helper_lib for neural network projects |
| 2025-12-12 | Added word embedding endpoints (HW1) |
| 2025-09-18 | Initial FastAPI setup with BigramModel |

---

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- FastAPI
- spaCy (for embedding endpoints)
- Pillow (for image processing)

