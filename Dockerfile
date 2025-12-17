# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set HuggingFace cache directory within the project codebase
# Models cached in: models/huggingface_cache/
# Datasets cached in: data/huggingface_datasets/
# Note: HF_HOME is the main cache dir (TRANSFORMERS_CACHE is deprecated in v5+)
ENV HF_HOME=/code/models/huggingface_cache
ENV HF_DATASETS_CACHE=/code/data/huggingface_datasets

# Set the working directory
WORKDIR /code

# Copy the pyproject.toml and uv.lock files
COPY pyproject.toml uv.lock /code/

# Install dependencies using uv (includes transformers, datasets, accelerate, huggingface_hub for LLM/RL)
RUN uv sync --frozen

# Install pip and download spaCy language model for word embeddings
RUN uv pip install pip && uv run python -m spacy download en_core_web_lg

# Create cache directories within the project
RUN mkdir -p /code/models/huggingface_cache /code/data/huggingface_datasets

# Pre-download the GPT2 model from HuggingFace to project cache (avoids download at runtime)
# Used by both Module 9 (llm_model.py) and Assignment 5 (llm_rl_model.py)
RUN uv run python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    cache_dir='/code/models/huggingface_cache'; \
    AutoTokenizer.from_pretrained('openai-community/gpt2', cache_dir=cache_dir); \
    AutoModelForCausalLM.from_pretrained('openai-community/gpt2', cache_dir=cache_dir)"

# Copy the application code
COPY ./app /code/app

# Copy the helper library for VAE and GAN models
COPY ./helper_lib /code/helper_lib

# Copy training scripts (for LLM fine-tuning and RL training)
COPY ./scripts /code/scripts

# Copy trained model weights (CNN, GAN, RNN, Diffusion, EBM, LLM, LLM-RL)
COPY ./models /code/models

# ============== Docker Usage Instructions ==============
# 
# Build the image:
#   docker build -t hello-world-genai .
#
# Run the API (standard):
#   docker run -p 8000:8000 hello-world-genai
#
# Run with volume mount (to persist trained models locally):
#   docker run -v $(pwd)/models:/code/models -p 8000:8000 hello-world-genai
#
# Run RL training inside container (Assignment 5):
#   docker run -it -v $(pwd)/models:/code/models hello-world-genai \
#     uv run python scripts/train_llm_rl.py --epochs 3 --num_samples 500
#
# Run RL training and upload to HuggingFace Hub:
#   docker run -it -v $(pwd)/models:/code/models \
#     -e HF_TOKEN=your_huggingface_token \
#     hello-world-genai \
#     uv run python scripts/train_llm_rl.py --epochs 3 --upload_to_hub "username/gpt2-squad-rl"
#
# ==============================================================

# Command to run the application on internal port 80
# CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]

# Command to run the application on docker container internal port 8000
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "8000"]