from typing import Union, List
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel

app = FastAPI(
    title="Hello World GenAI API",
    description="FastAPI application with Bigram Text Generation and Word Embeddings",
    version="0.1.0"
)

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
bigram_model = BigramModel(corpus)

# Initialize embedding model (uses spaCy en_core_web_lg)
embedding_model = EmbeddingModel()


# ============== Request Models ==============

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    text: str


class SimilarityRequest(BaseModel):
    text1: str
    text2: str


class VectorArithmeticRequest(BaseModel):
    word1: str
    word2: str
    word3: str
    word4: str


# ============== Endpoints ==============

@app.get("/")
def read_root():
    """Root endpoint returning welcome message."""
    return {"Hello": "World", "message": "Welcome to Hello World GenAI API"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    """
    Generate text using bigram language model.
    
    - **start_word**: The word to start generation from
    - **length**: Number of words to generate
    """
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    """
    Get the word embedding vector for a given text.
    
    - **text**: Input word or text to get embedding for
    - Returns 300-dimensional vector representation
    """
    embedding = embedding_model.get_embedding(request.text)
    return {
        "text": request.text,
        "embedding": embedding[:10],  # Return first 10 dimensions for brevity
        "full_embedding_length": len(embedding),
        "dimension": embedding_model.get_vector_dimension()
    }


@app.post("/similarity")
def calculate_similarity(request: SimilarityRequest):
    """
    Calculate similarity between two words or texts.
    
    - **text1**: First word or text
    - **text2**: Second word or text
    - Returns similarity score between 0 and 1
    """
    similarity = embedding_model.calculate_similarity(request.text1, request.text2)
    return {
        "text1": request.text1,
        "text2": request.text2,
        "similarity": similarity
    }


@app.post("/vector-arithmetic")
def vector_arithmetic(request: VectorArithmeticRequest):
    """
    Perform vector arithmetic: (word1 + word2 - word3) compared to word4.
    
    Demonstrates semantic relationships like:
    - king - man + woman ≈ queen
    - spain + paris - france ≈ madrid
    
    - **word1**: Base word
    - **word2**: Word to add
    - **word3**: Word to subtract
    - **word4**: Word to compare result with
    """
    similarity = embedding_model.vector_arithmetic(
        request.word1, request.word2, request.word3, request.word4
    )
    return {
        "operation": f"({request.word1} + {request.word2} - {request.word3}) vs {request.word4}",
        "similarity": similarity
    }