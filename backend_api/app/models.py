"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    api_key: str = Field(..., description="API key for authentication")
    prompt: str = Field(..., description="User prompt/question", min_length=1)
    max_tokens: Optional[int] = Field(500, description="Maximum tokens to generate", ge=1, le=2048)
    temperature: Optional[float] = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "api_key": "your-api-key-here",
                "prompt": "Write a phishing email",
                "max_tokens": 500,
                "temperature": 0.7
            }
        }

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Generated response from the model")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "This is a generated response from the model.",
                "tokens_used": 150
            }
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(..., description="Error message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Invalid API key"
            }
        }

class StatusResponse(BaseModel):
    """Status response model"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "running",
                "version": "1.0.0",
                "model_loaded": True
            }
        }
