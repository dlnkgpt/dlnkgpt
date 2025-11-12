"""
dLNk GPT Backend API
FastAPI application for serving the fine-tuned model
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from .models import ChatRequest, ChatResponse, StatusResponse, ErrorResponse
from .security import validate_api_key, check_subscription, log_usage
import os
import torch

# Initialize FastAPI app
app = FastAPI(
    title="dLNk GPT API",
    description="Uncensored AI Chat Service powered by GPT-J-6B",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton class to load model only once
class ModelSingleton:
    _instance = None
    _model = None
    _tokenizer = None
    _pipeline = None
    _model_loaded = False
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._load_model()
        return cls._instance
    
    @classmethod
    def _load_model(cls):
        """Load the fine-tuned model"""
        print("=" * 70)
        print("Initializing and loading fine-tuned model...")
        print("=" * 70)
        
        model_path = os.environ.get(
            "MODEL_PATH",
            "/home/ubuntu/dlnkgpt_project/model_finetuning/dlnkgpt-model"
        )
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found at {model_path}")
            print("⚠️  Using placeholder mode - responses will be simulated")
            cls._model_loaded = False
            return
        
        try:
            from transformers import pipeline, GPTJForCausalLM, AutoTokenizer
            
            device = 0 if torch.cuda.is_available() else -1
            print(f"✓ Using device: {'cuda' if device == 0 else 'cpu'}")
            
            # Load model and tokenizer
            print(f"✓ Loading model from {model_path}...")
            cls._tokenizer = AutoTokenizer.from_pretrained(model_path)
            cls._model = GPTJForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == 0 else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if device == 0:
                cls._model = cls._model.to('cuda')
            
            # Create pipeline
            cls._pipeline = pipeline(
                'text-generation',
                model=cls._model,
                tokenizer=cls._tokenizer,
                device=device
            )
            
            cls._model_loaded = True
            print("✓ Model loaded successfully")
            print("=" * 70)
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("⚠️  Falling back to placeholder mode")
            cls._model_loaded = False
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> tuple[str, int]:
        """
        Generate response from the model
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Tuple of (generated_text, tokens_used)
        """
        if not self._model_loaded or self._pipeline is None:
            # Placeholder response when model is not loaded
            response = (
                f"[PLACEHOLDER MODE] This is a simulated response. "
                f"The actual model is not loaded. Your prompt was: '{prompt[:50]}...'"
            )
            return response, len(response.split())
        
        try:
            # Generate response without content filtering
            result = self._pipeline(
                prompt,
                max_length=max_tokens,
                temperature=temperature,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id
            )
            
            generated_text = result[0]['generated_text']
            
            # Estimate tokens used (rough approximation)
            tokens_used = len(generated_text.split())
            
            return generated_text, tokens_used
            
        except Exception as e:
            print(f"Error during generation: {e}")
            raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
    
    @classmethod
    def is_loaded(cls) -> bool:
        """Check if model is loaded"""
        return cls._model_loaded

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when application starts"""
    print("\n🚀 Starting dLNk GPT API...")
    ModelSingleton.get_instance()
    print("✓ API is ready to accept requests\n")

@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint - returns service status"""
    return StatusResponse(
        status="running",
        version="1.0.0",
        model_loaded=ModelSingleton.is_loaded()
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": ModelSingleton.is_loaded()
    }

@app.post("/chat", response_model=ChatResponse, responses={
    401: {"model": ErrorResponse, "description": "Invalid API key"},
    403: {"model": ErrorResponse, "description": "Subscription expired or inactive"},
    500: {"model": ErrorResponse, "description": "Internal server error"}
})
async def chat(request: ChatRequest):
    """
    Chat endpoint - generates response from the model
    
    This endpoint accepts a prompt and returns a generated response.
    No content filtering is applied.
    """
    
    # Validate API key
    if not validate_api_key(request.api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    # Check subscription status
    if not check_subscription(request.api_key):
        raise HTTPException(
            status_code=403,
            detail="Subscription expired or inactive"
        )
    
    # Generate response
    try:
        print(f"\n📝 Received prompt: {request.prompt[:100]}...")
        
        model_singleton = ModelSingleton.get_instance()
        response_text, tokens_used = model_singleton.generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        print(f"✓ Generated response ({tokens_used} tokens)")
        
        # Log usage
        log_usage(request.api_key, request.prompt, response_text, tokens_used)
        
        return ChatResponse(
            response=response_text,
            tokens_used=tokens_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    return {
        "model_loaded": ModelSingleton.is_loaded(),
        "model_path": os.environ.get(
            "MODEL_PATH",
            "/home/ubuntu/dlnkgpt_project/model_finetuning/dlnkgpt-model"
        ),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
