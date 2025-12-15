# Hello World GenAI as HW1 codebase

A FastAPI application demonstrating bigram-based text generation and word embeddings for APAN 5560 Applied Generative AI course.

## Features

- **Bigram Language Model**: Generates text using bigram probabilities learned from a corpus
- **Word Embeddings**: Extract 300-dimensional word vectors using spaCy
- **Similarity Calculation**: Compute semantic similarity between words/sentences
- **Vector Arithmetic**: Demonstrate semantic relationships (e.g., king - man + woman ≈ queen)
- **FastAPI REST API**: Exposes all functionality via HTTP endpoints

## Project Structure

```
hello_world_genai/
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI application entry point
│   ├── bigram_model.py     # BigramModel class for text generation
│   └── embedding_model.py  # EmbeddingModel class for word embeddings
├── pyproject.toml          # Project dependencies
├── Dockerfile              # Docker configuration
└── README.md
```

## Setup

### Install Dependencies

```bash
cd hello_world_genai
uv sync
```

### Download spaCy Model

```bash
python -m spacy download en_core_web_lg
```

## API Endpoints

### GET /
Returns a hello world message.

### POST /generate
Generates text using the bigram model.

**Request:**
```json
{
  "start_word": "the",
  "length": 10
}
```

**Response:**
```json
{
  "generated_text": "the count of monte cristo is a novel..."
}
```

### POST /embedding
Get the word embedding vector for a given text.

**Request:**
```json
{
  "text": "apple"
}
```

**Response:**
```json
{
  "text": "apple",
  "embedding": [0.23, -0.45, ...],
  "full_embedding_length": 300,
  "dimension": 300
}
```

### POST /similarity
Calculate similarity between two words or texts.

**Request:**
```json
{
  "text1": "apple",
  "text2": "orange"
}
```

**Response:**
```json
{
  "text1": "apple",
  "text2": "orange",
  "similarity": 0.56
}
```

### POST /vector-arithmetic
Perform vector arithmetic to demonstrate semantic relationships.

**Request:**
```json
{
  "word1": "spain",
  "word2": "paris",
  "word3": "france",
  "word4": "madrid"
}
```

**Response:**
```json
{
  "operation": "(spain + paris - france) vs madrid",
  "similarity": 0.78
}
```

## Running the Application

### Development Mode

```bash
cd hello_world_genai
uv run fastapi dev app/main.py
```

### Production Mode (Docker)

```bash
docker build -t hello_world_genai .
docker run -p 8000:8000 hello_world_genai
```

Then for development mode at localhost,
 visit:
- http://127.0.0.1:8000 - API root
- http://127.0.0.1:8000/docs - Interactive API documentation (Swagger UI)

for docker deployment, 
visit:     
server   Server started at http://0.0.0.0:8000
server   Documentation at http://0.0.0.0:8000/docs

