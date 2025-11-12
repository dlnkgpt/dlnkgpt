"""
Security and authentication functions
"""

import hashlib
from datetime import datetime
from typing import Optional

# In-memory storage for demo purposes
# In production, use a proper database
VALID_API_KEYS = {
    "demo_key_123": {
        "user_id": 1,
        "username": "demo_user",
        "subscription_tier": "premium",
        "is_active": True,
        "expires_at": None
    },
    "test_key_456": {
        "user_id": 2,
        "username": "test_user",
        "subscription_tier": "basic",
        "is_active": True,
        "expires_at": None
    }
}

def hash_api_key(api_key: str) -> str:
    """Hash an API key using SHA256"""
    return hashlib.sha256(api_key.encode()).hexdigest()

def validate_api_key(api_key: str) -> bool:
    """
    Validate if the provided API key is valid
    
    Args:
        api_key: The API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not api_key:
        return False
    
    # Check if key exists in our storage
    if api_key in VALID_API_KEYS:
        key_info = VALID_API_KEYS[api_key]
        
        # Check if key is active
        if not key_info.get("is_active", False):
            return False
        
        # Check expiration
        expires_at = key_info.get("expires_at")
        if expires_at and datetime.now() > expires_at:
            return False
        
        return True
    
    return False

def check_subscription(api_key: str) -> bool:
    """
    Check if the API key has an active subscription
    
    Args:
        api_key: The API key to check
        
    Returns:
        True if subscription is active, False otherwise
    """
    if api_key not in VALID_API_KEYS:
        return False
    
    key_info = VALID_API_KEYS[api_key]
    
    # Check if subscription is active
    if not key_info.get("is_active", False):
        return False
    
    # Check subscription tier
    tier = key_info.get("subscription_tier", "")
    if tier not in ["basic", "premium", "enterprise"]:
        return False
    
    return True

def get_user_info(api_key: str) -> Optional[dict]:
    """
    Get user information associated with an API key
    
    Args:
        api_key: The API key
        
    Returns:
        User info dict or None if not found
    """
    return VALID_API_KEYS.get(api_key)

def log_usage(api_key: str, prompt: str, response: str, tokens_used: int = 0):
    """
    Log API usage for billing and analytics
    
    Args:
        api_key: The API key used
        prompt: The input prompt
        response: The generated response
        tokens_used: Number of tokens used
    """
    # In production, save to database
    user_info = get_user_info(api_key)
    if user_info:
        print(f"[USAGE LOG] User: {user_info['username']} | "
              f"Tokens: {tokens_used} | "
              f"Prompt length: {len(prompt)} | "
              f"Response length: {len(response)}")

def create_api_key(username: str, subscription_tier: str = "basic") -> str:
    """
    Create a new API key for a user
    
    Args:
        username: Username
        subscription_tier: Subscription tier (basic, premium, enterprise)
        
    Returns:
        The generated API key
    """
    import secrets
    
    # Generate a secure random API key
    api_key = f"dlnk_{secrets.token_urlsafe(32)}"
    
    # Store in our in-memory database
    VALID_API_KEYS[api_key] = {
        "user_id": len(VALID_API_KEYS) + 1,
        "username": username,
        "subscription_tier": subscription_tier,
        "is_active": True,
        "expires_at": None
    }
    
    return api_key
