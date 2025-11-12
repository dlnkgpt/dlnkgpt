"""
dLNk GPT Backend API - Advanced Version with Database Integration
FastAPI application for serving the fine-tuned uncensored model
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from .models import ChatRequest, ChatResponse, StatusResponse, ErrorResponse
from .database import get_db, init_db
from .auth import AuthManager, get_current_user, require_active_subscription
import os
import torch
import time
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="dLNk GPT API - Advanced",
    description="Uncensored AI Chat Service powered by Fine-tuned GPT-J-6B",
    version="2.0.0",
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
        """Load the fine-tuned uncensored model"""
        print("=" * 70)
        print("Initializing and loading fine-tuned uncensored model...")
        print("=" * 70)
        
        model_path = os.environ.get(
            "MODEL_PATH",
            "/home/ubuntu/dlnkgpt/model_finetuning/dlnkgpt-uncensored-model"
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
            if cls._tokenizer.pad_token is None:
                cls._tokenizer.pad_token = cls._tokenizer.eos_token
            
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
            print("✓ Uncensored model loaded successfully")
            print("=" * 70)
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("⚠️  Falling back to placeholder mode")
            cls._model_loaded = False
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> tuple[str, int, float]:
        """
        Generate response from the model
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Tuple of (generated_text, tokens_used, generation_time)
        """
        start_time = time.time()
        
        if not self._model_loaded or self._pipeline is None:
            # Placeholder response when model is not loaded
            response = (
                f"[PLACEHOLDER MODE] This is a simulated uncensored response. "
                f"The actual model is not loaded. Your prompt was: '{prompt[:50]}...'"
            )
            generation_time = time.time() - start_time
            return response, len(response.split()), generation_time
        
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
            
            # Estimate tokens used
            tokens_used = len(generated_text.split())
            
            generation_time = time.time() - start_time
            
            return generated_text, tokens_used, generation_time
            
        except Exception as e:
            print(f"Error during generation: {e}")
            raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
    
    @classmethod
    def is_loaded(cls) -> bool:
        """Check if model is loaded"""
        return cls._model_loaded

# Initialize database and model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and load model when application starts"""
    print("\n🚀 Starting dLNk GPT Advanced API...")
    
    # Initialize database
    print("\n[*] Initializing database...")
    init_db()
    print("✓ Database ready")
    
    # Load model
    ModelSingleton.get_instance()
    print("✓ API is ready to accept requests\n")

@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint - returns service status"""
    return StatusResponse(
        status="running",
        version="2.0.0",
        model_loaded=ModelSingleton.is_loaded()
    )

@app.get("/health")
async def health(db: Session = Depends(get_db)):
    """Health check endpoint"""
    # Check database connection
    try:
        from .database import User
        db.query(User).first()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "model_loaded": ModelSingleton.is_loaded(),
        "database": db_status
    }

@app.post("/chat", response_model=ChatResponse, responses={
    401: {"model": ErrorResponse, "description": "Invalid API key"},
    403: {"model": ErrorResponse, "description": "Subscription expired or rate limit exceeded"},
    500: {"model": ErrorResponse, "description": "Internal server error"}
})
async def chat(
    request: ChatRequest,
    http_request: Request,
    db: Session = Depends(get_db)
):
    """
    Chat endpoint - generates uncensored response from the model
    
    This endpoint accepts a prompt and returns a generated response.
    No content filtering is applied.
    """
    
    # Validate API key
    api_key_record = AuthManager.validate_api_key(db, request.api_key)
    if not api_key_record:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    # Check rate limit
    allowed, reason = AuthManager.check_rate_limit(db, api_key_record.id)
    if not allowed:
        raise HTTPException(
            status_code=403,
            detail=reason
        )
    
    # Check subscription status
    has_subscription, subscription = AuthManager.check_subscription(db, api_key_record.user_id)
    if not has_subscription:
        raise HTTPException(
            status_code=403,
            detail="Subscription expired or inactive"
        )
    
    # Generate response
    try:
        print(f"\n📝 Received prompt from user {api_key_record.user_id}: {request.prompt[:100]}...")
        
        model_singleton = ModelSingleton.get_instance()
        response_text, tokens_used, generation_time = model_singleton.generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        print(f"✓ Generated response ({tokens_used} tokens, {generation_time:.2f}s)")
        
        # Increment rate limit
        AuthManager.increment_rate_limit(db, api_key_record.id, tokens_used)
        
        # Log usage
        AuthManager.log_usage(
            db=db,
            user_id=api_key_record.user_id,
            api_key_id=api_key_record.id,
            endpoint="/chat",
            method="POST",
            prompt=request.prompt,
            response=response_text,
            tokens_used=tokens_used,
            generation_time=generation_time,
            status_code=200,
            ip_address=http_request.client.host if http_request.client else None,
            user_agent=http_request.headers.get("user-agent")
        )
        
        return ChatResponse(
            response=response_text,
            tokens_used=tokens_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error: {e}")
        
        # Log error
        AuthManager.log_usage(
            db=db,
            user_id=api_key_record.user_id,
            api_key_id=api_key_record.id,
            endpoint="/chat",
            method="POST",
            prompt=request.prompt,
            response=None,
            tokens_used=0,
            status_code=500,
            error_message=str(e),
            ip_address=http_request.client.host if http_request.client else None,
            user_agent=http_request.headers.get("user-agent")
        )
        
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
            "/home/ubuntu/dlnkgpt/model_finetuning/dlnkgpt-uncensored-model"
        ),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "version": "2.0.0",
        "features": [
            "uncensored_responses",
            "adversarial_training",
            "safety_layer_unlocking",
            "database_integration",
            "rate_limiting",
            "usage_tracking"
        ]
    }

@app.get("/user/stats")
async def user_stats(
    api_key: str,
    db: Session = Depends(get_db)
):
    """Get user statistics"""
    
    # Validate API key
    api_key_record = AuthManager.validate_api_key(db, api_key)
    if not api_key_record:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    # Get stats
    stats = AuthManager.get_user_stats(db, api_key_record.user_id)
    
    return stats

@app.get("/user/api-keys")
async def list_api_keys(
    api_key: str,
    db: Session = Depends(get_db)
):
    """List user's API keys"""
    
    # Validate API key
    api_key_record = AuthManager.validate_api_key(db, api_key)
    if not api_key_record:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    # Get user's API keys
    from .database import APIKey
    api_keys = db.query(APIKey).filter(
        APIKey.user_id == api_key_record.user_id
    ).all()
    
    return {
        "api_keys": [
            {
                "id": key.id,
                "name": key.name,
                "is_active": key.is_active,
                "created_at": key.created_at.isoformat(),
                "last_used": key.last_used.isoformat() if key.last_used else None,
                "expires_at": key.expires_at.isoformat() if key.expires_at else None
            }
            for key in api_keys
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main_advanced:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
