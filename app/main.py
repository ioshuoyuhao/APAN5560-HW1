from typing import Union, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.classifier_model import ImageClassifier
from app.gan_model import MNISTGANGenerator
from app.rnn_model import RNNTextGenerator
from app.diffusion_model import DiffusionImageGenerator
from app.ebm_model import EBMImageGenerator
from app.llm_model import LLMTextGenerator
from app.llm_rl_model import LLMRLGenerator
import base64

app = FastAPI(
    title="Hello World GenAI API",
    description="FastAPI application with Bigram/RNN/LLM/LLM-RL Text Generation, Word Embeddings, Image Classification, MNIST GAN, Diffusion, and EBM (MNIST + CIFAR-10)",
    version="0.5.0"  # HW5 - Added RL-based LLM fine-tuning with GPT2 for Q&A using PPO
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

# Initialize image classifier (Assignment 2 - CNN for CIFAR-10)
# Model path can be set to load trained weights
classifier = ImageClassifier(model_path="models/assignment_cnn.pth")

# Initialize MNIST GAN generator (Assignment 3 - GAN for handwritten digits)
# Model path can be set to load trained generator weights
gan_generator = MNISTGANGenerator(model_path="models/mnist_gan_generator.pth", z_dim=100)

# Initialize RNN (LSTM) text generator (Module 7 - RNN for text generation)
# Try to load pre-trained model, otherwise train and save
rnn_generator = RNNTextGenerator(vocab_size=10000, embedding_dim=100, hidden_dim=128, seq_len=30)
if not rnn_generator.load_model("models/rnn_model.pth"):
    print("No pre-trained RNN model found. Training from scratch...")
    print("Training RNN model on Count of Monte Cristo text...")
    rnn_generator.train(epochs=15, batch_size=64)
    rnn_generator.save_model("models/rnn_model.pth")
    print("RNN model training complete and saved!")

# Initialize Diffusion image generator (HW4 - Diffusion model for image generation)
# Model path can be set to load trained weights
diffusion_generator = DiffusionImageGenerator(
    image_size=64, 
    num_channels=3,
    model_path="models/diffusion_checkpoints/diffusion_best.pth"
)

# Initialize EBM image generator for CIFAR-10 (HW4 - Energy-Based Model for RGB image generation)
# Model path can be set to load trained weights
ebm_generator = EBMImageGenerator(
    image_size=32,
    num_channels=3,  # RGB for CIFAR-10
    model_path="models/ebm_cifar10_best.pth"
)

# Initialize LLM (GPT2) text generator (Module 9 - Fine-tuned GPT2 for Q&A)
# Model path can be set to load fine-tuned weights
llm_generator = LLMTextGenerator(
    model_name="openai-community/gpt2",
    model_path="models/llm_finetuned.pth",
    max_length=128
)

# Initialize LLM-RL (GPT2 with RL/PPO) text generator (HW5 - RL fine-tuned GPT2 for Q&A)
# Uses SQuAD dataset and shaped rewards for response format
# Priority: local model_path > HuggingFace Hub > pretrained base model
# Public HF repos don't require authentication token for downloading
llm_rl_generator = LLMRLGenerator(
    model_name="openai-community/gpt2",
    model_path="models/llm_rl_finetuned.pth",
    max_length=256,
    hf_repo="StevenHuo/StevenHuo-gpt2-squad-rl"  # HW5: Public HF Hub repo
)


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


class GANGenerateRequest(BaseModel):
    num_samples: int = 1


class RNNTextGenerationRequest(BaseModel):
    start_word: str
    length: int = 50
    temperature: float = 1.0


class DiffusionGenerateRequest(BaseModel):
    num_samples: int = 1
    diffusion_steps: int = 100


class EBMGenerateRequest(BaseModel):
    num_samples: int = 1
    steps: int = 256
    step_size: float = 10.0
    noise_std: float = 0.01


class LLMTextGenerationRequest(BaseModel):
    """Request model for LLM text generation endpoint."""
    start_word: str  # The prompt/question to generate from
    length: int = 50  # Number of new tokens to generate
    temperature: float = 1.0  # Sampling temperature (higher = more random)
    top_k: int = 50  # Top-k sampling parameter
    top_p: float = 0.95  # Top-p (nucleus) sampling parameter


class LLMRLTextGenerationRequest(BaseModel):
    """Request model for RL-trained LLM text generation endpoint (HW5)."""
    question: str  # The question to answer
    context: str = ""  # Optional context for the question
    max_tokens: int = 100  # Number of new tokens to generate
    temperature: float = 0.8  # Sampling temperature


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


@app.post("/generate_with_rnn")
def generate_with_rnn(request: RNNTextGenerationRequest):
    """
    Generate text using LSTM (RNN) language model.
    
    The model is trained on Count of Monte Cristo text and uses LSTM
    architecture for more coherent text generation compared to bigram model.
    
    - **start_word**: The word or phrase to start generation from
    - **length**: Number of words to generate (default: 50)
    - **temperature**: Sampling temperature (default: 1.0, higher = more random)
    """
    try:
        generated_text = rnn_generator.generate_text(
            seed_text=request.start_word,
            length=request.length,
            temperature=request.temperature
        )
        return {"generated_text": generated_text}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_with_llm")
def generate_with_llm(request: LLMTextGenerationRequest):
    """
    Generate text using fine-tuned GPT2 (LLM) language model.
    
    The model is fine-tuned on the Nectar Q&A dataset from HuggingFace
    and uses Transformer architecture for question-answering text generation.
    
    Reference: https://huggingface.co/docs/transformers/en/index
    
    - **start_word**: The prompt/question to generate from
    - **length**: Number of new tokens to generate (default: 50)
    - **temperature**: Sampling temperature (default: 1.0, higher = more random)
    - **top_k**: Top-k sampling parameter (default: 50)
    - **top_p**: Top-p (nucleus) sampling parameter (default: 0.95)
    """
    try:
        generated_text = llm_generator.generate_text(
            prompt=request.start_word,
            max_new_tokens=request.length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )
        return {"generated_text": generated_text}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm/info")
def get_llm_info():
    """
    Get information about the LLM (GPT2) model.
    """
    return llm_generator.get_model_info()


# ============== RL-trained LLM Endpoints (Assignment 5) ==============

@app.post("/generate_with_llm_rl")
def generate_with_llm_rl(request: LLMRLTextGenerationRequest):
    """
    Generate text using RL-trained GPT2 (LLM) language model.
    
    HW5: The model is fine-tuned using Reinforcement Learning (PPO) on the 
    SQuAD dataset. It generates responses with a specific format:
    - Starts with: "That is a great question! "
    - Ends with: " Let me know if you have any other questions."
    
    Training Method: PPO (Proximal Policy Optimization) from Module 10 & 11
    Dataset: SQuAD (https://huggingface.co/datasets/rajpurkar/squad)
    
    - **question**: The question to answer
    - **context**: Optional context for the question
    - **max_tokens**: Number of new tokens to generate (default: 100)
    - **temperature**: Sampling temperature (default: 0.8)
    """
    try:
        generated_text = llm_rl_generator.generate_text(
            question=request.question,
            context=request.context,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return {"generated_text": generated_text}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm_rl/info")
def get_llm_rl_info():
    """
    Get information about the RL-trained LLM (GPT2 with PPO) model.
    
    HW5: Displays model info including training method, response format,
    and dataset used for fine-tuning.
    """
    return llm_rl_generator.get_model_info()


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


# ============== Image Classification Endpoints (Assignment 2) ==============

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify an image using the CIFAR-10 CNN model.
    
    - **file**: Image file (JPEG, PNG, etc.) to classify
    - Returns predicted class, confidence, and top 5 predictions
    
    CIFAR-10 Classes: airplane, automobile, bird, cat, deer, 
                      dog, frog, horse, ship, truck
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Classify the image
        predicted_class, confidence, top5 = classifier.classify(image_bytes)
        
        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "top5_predictions": [
                {"class": cls, "probability": round(prob, 4)}
                for cls, prob in top5
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/classify/classes")
def get_classes():
    """
    Get the list of CIFAR-10 class names that the model can predict.
    """
    return {
        "classes": classifier.get_class_names(),
        "num_classes": len(classifier.get_class_names())
    }


# ============== MNIST GAN Endpoints (Assignment 3) ==============

@app.post("/gan/generate")
def generate_mnist_digits(request: GANGenerateRequest):
    """
    Generate handwritten digit images using the trained MNIST GAN.
    
    - **num_samples**: Number of images to generate (1-64)
    - Returns base64-encoded PNG images
    """
    # Validate num_samples
    if request.num_samples < 1 or request.num_samples > 64:
        raise HTTPException(
            status_code=400, 
            detail="num_samples must be between 1 and 64"
        )
    
    try:
        # Generate images as base64
        images_base64 = gan_generator.generate_base64(request.num_samples)
        
        return {
            "num_samples": request.num_samples,
            "latent_dim": gan_generator.get_latent_dim(),
            "images": images_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating images: {str(e)}")


@app.get("/gan/generate/grid")
def generate_mnist_grid(
    num_samples: int = Query(default=16, ge=1, le=64, description="Number of images to generate"),
    nrow: int = Query(default=4, ge=1, le=8, description="Number of images per row")
):
    """
    Generate a grid of handwritten digit images using the trained MNIST GAN.
    
    - **num_samples**: Number of images to generate (1-64)
    - **nrow**: Number of images per row in the grid (1-8)
    - Returns a single base64-encoded PNG image of the grid
    """
    try:
        # Generate grid image as base64
        grid_base64 = gan_generator.generate_grid(num_samples, nrow)
        
        return {
            "num_samples": num_samples,
            "nrow": nrow,
            "latent_dim": gan_generator.get_latent_dim(),
            "grid_image": grid_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating grid: {str(e)}")


@app.get("/gan/generate/image")
def generate_single_mnist_image():
    """
    Generate a single handwritten digit image and return as PNG.
    
    Returns a PNG image directly (can be viewed in browser).
    """
    try:
        # Generate single image as base64
        images_base64 = gan_generator.generate_base64(1)
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(images_base64[0])
        
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")


@app.get("/gan/info")
def get_gan_info():
    """
    Get information about the MNIST GAN model.
    """
    return {
        "model": "MNIST GAN",
        "architecture": {
            "generator": {
                "input": "Noise vector (z_dim)",
                "layers": [
                    "Linear: z_dim → 7×7×128",
                    "Reshape to (128, 7, 7)",
                    "ConvTranspose2d: 128 → 64, k=4, s=2, p=1 + BatchNorm + ReLU → 14×14",
                    "ConvTranspose2d: 64 → 1, k=4, s=2, p=1 + Tanh → 28×28"
                ],
                "output": "28×28 grayscale image"
            },
            "discriminator": {
                "input": "28×28 grayscale image",
                "layers": [
                    "Conv2d: 1 → 64, k=4, s=2, p=1 + LeakyReLU(0.2) → 14×14",
                    "Conv2d: 64 → 128, k=4, s=2, p=1 + BatchNorm + LeakyReLU(0.2) → 7×7",
                    "Flatten + Linear: 6272 → 1"
                ],
                "output": "Real/Fake probability"
            }
        },
        "latent_dim": gan_generator.get_latent_dim(),
        "output_shape": "(1, 28, 28)",
        "training_dataset": "MNIST"
    }


# ============== Diffusion Model Endpoints (HW4) ==============

@app.post("/diffusion/generate")
def generate_diffusion_images(request: DiffusionGenerateRequest):
    """
    Generate images using the trained Diffusion model.
    
    - **num_samples**: Number of images to generate (1-16)
    - **diffusion_steps**: Number of diffusion steps (more = better quality, slower)
    - Returns base64-encoded PNG images
    """
    # Validate num_samples
    if request.num_samples < 1 or request.num_samples > 16:
        raise HTTPException(
            status_code=400,
            detail="num_samples must be between 1 and 16"
        )
    
    try:
        images_base64 = diffusion_generator.generate_base64(
            request.num_samples, 
            request.diffusion_steps
        )
        
        return {
            "num_samples": request.num_samples,
            "diffusion_steps": request.diffusion_steps,
            "images": images_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating images: {str(e)}")


@app.get("/diffusion/generate/grid")
def generate_diffusion_grid(
    num_samples: int = Query(default=9, ge=1, le=16, description="Number of images to generate"),
    nrow: int = Query(default=3, ge=1, le=4, description="Number of images per row"),
    diffusion_steps: int = Query(default=100, ge=10, le=1000, description="Number of diffusion steps")
):
    """
    Generate a grid of images using the Diffusion model.
    
    - **num_samples**: Number of images to generate (1-16)
    - **nrow**: Number of images per row in the grid (1-4)
    - **diffusion_steps**: Number of diffusion steps
    - Returns a single base64-encoded PNG image of the grid
    """
    try:
        grid_base64 = diffusion_generator.generate_grid(num_samples, nrow, diffusion_steps)
        
        return {
            "num_samples": num_samples,
            "nrow": nrow,
            "diffusion_steps": diffusion_steps,
            "grid_image": grid_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating grid: {str(e)}")


@app.get("/diffusion/generate/image")
def generate_single_diffusion_image(
    diffusion_steps: int = Query(default=100, ge=10, le=1000, description="Number of diffusion steps")
):
    """
    Generate a single image using the Diffusion model and return as PNG.
    
    Returns a PNG image directly (can be viewed in browser).
    """
    try:
        images_base64 = diffusion_generator.generate_base64(1, diffusion_steps)
        image_bytes = base64.b64decode(images_base64[0])
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")


@app.get("/diffusion/info")
def get_diffusion_info():
    """
    Get information about the Diffusion model.
    """
    return diffusion_generator.get_info()


# ============== Energy-Based Model (EBM) Endpoints - CIFAR-10 (HW4) ==============

@app.post("/ebm/generate")
def generate_ebm_images(request: EBMGenerateRequest):
    """
    Generate RGB images using the CIFAR-10 trained Energy-Based Model (EBM).
    
    Uses Langevin dynamics to sample low-energy states.
    Generates 32x32 RGB images.
    
    - **num_samples**: Number of images to generate (1-16)
    - **steps**: Number of Langevin sampling steps (more = better quality)
    - **step_size**: Langevin step size
    - **noise_std**: Noise standard deviation for sampling
    - Returns base64-encoded PNG images
    """
    # Validate num_samples
    if request.num_samples < 1 or request.num_samples > 16:
        raise HTTPException(
            status_code=400,
            detail="num_samples must be between 1 and 16"
        )
    
    try:
        images_base64 = ebm_generator.generate_base64(
            request.num_samples,
            request.steps,
            request.step_size,
            request.noise_std
        )
        
        return {
            "num_samples": request.num_samples,
            "langevin_steps": request.steps,
            "step_size": request.step_size,
            "noise_std": request.noise_std,
            "dataset": "CIFAR-10 (RGB)",
            "images": images_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating images: {str(e)}")


@app.get("/ebm/generate/grid")
def generate_ebm_grid(
    num_samples: int = Query(default=9, ge=1, le=16, description="Number of images to generate"),
    nrow: int = Query(default=3, ge=1, le=4, description="Number of images per row"),
    steps: int = Query(default=256, ge=10, le=1000, description="Number of Langevin steps")
):
    """
    Generate a grid of RGB images using the CIFAR-10 EBM.
    
    - **num_samples**: Number of images to generate (1-16)
    - **nrow**: Number of images per row in the grid (1-4)
    - **steps**: Number of Langevin sampling steps
    - Returns a single base64-encoded PNG image of the grid
    """
    try:
        grid_base64 = ebm_generator.generate_grid(num_samples, nrow, steps)
        
        return {
            "num_samples": num_samples,
            "nrow": nrow,
            "langevin_steps": steps,
            "dataset": "CIFAR-10 (RGB)",
            "grid_image": grid_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating grid: {str(e)}")


@app.get("/ebm/generate/image")
def generate_single_ebm_image(
    steps: int = Query(default=256, ge=10, le=1000, description="Number of Langevin steps")
):
    """
    Generate a single RGB image using the CIFAR-10 EBM and return as PNG.
    
    Returns a PNG image directly (can be viewed in browser).
    """
    try:
        images_base64 = ebm_generator.generate_base64(1, steps)
        image_bytes = base64.b64decode(images_base64[0])
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")


@app.get("/ebm/info")
def get_ebm_info():
    """
    Get information about the CIFAR-10 Energy-Based Model (EBM).
    """
    return ebm_generator.get_info()