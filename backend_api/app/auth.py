"""
Authentication and user management for dLNk GPT
"""

from datetime import datetime, timedelta
from typing import Optional
import secrets
import hashlib
from sqlalchemy.orm import Session
import bcrypt

from .database import User, APIKey, Subscription, UsageLog, RateLimitTracker

class AuthManager:
    """
    Authentication and user management
    """
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using bcrypt
        """
        # Convert password to bytes and hash
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against hash
        """
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """
        Hash an API key using SHA256
        """
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    def generate_api_key() -> str:
        """
        Generate a secure API key
        """
        return f"dlnk_{secrets.token_urlsafe(32)}"
    
    @staticmethod
    def create_user(
        db: Session,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None
    ) -> User:
        """
        Create a new user
        """
        # Check if user exists
        existing_user = db.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            raise ValueError("Username or email already exists")
        
        # Create user
        hashed_password = AuthManager.hash_password(password)
        
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            is_active=True,
            is_verified=False,
            is_admin=False
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # Create default free subscription
        AuthManager.create_subscription(db, user.id, tier="free")
        
        return user
    
    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user
        """
        user = db.query(User).filter(User.username == username).first()
        
        if not user:
            return None
        
        if not AuthManager.verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        return user
    
    @staticmethod
    def create_api_key(
        db: Session,
        user_id: int,
        name: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        rate_limit_per_minute: int = 60,
        rate_limit_per_day: int = 10000
    ) -> APIKey:
        """
        Create a new API key for a user
        """
        # Generate API key
        api_key = AuthManager.generate_api_key()
        key_hash = AuthManager.hash_api_key(api_key)
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Create API key record
        api_key_record = APIKey(
            user_id=user_id,
            key=api_key,
            key_hash=key_hash,
            name=name or f"API Key {datetime.utcnow().strftime('%Y%m%d')}",
            is_active=True,
            expires_at=expires_at,
            rate_limit_per_minute=rate_limit_per_minute,
            rate_limit_per_day=rate_limit_per_day
        )
        
        db.add(api_key_record)
        db.commit()
        db.refresh(api_key_record)
        
        # Initialize rate limit tracker
        rate_tracker = RateLimitTracker(
            api_key_id=api_key_record.id,
            requests_per_minute=0,
            requests_per_day=0,
            tokens_per_day=0,
            minute_reset_at=datetime.utcnow() + timedelta(minutes=1),
            day_reset_at=datetime.utcnow() + timedelta(days=1)
        )
        
        db.add(rate_tracker)
        db.commit()
        
        return api_key_record
    
    @staticmethod
    def validate_api_key(db: Session, api_key: str) -> Optional[APIKey]:
        """
        Validate an API key
        """
        # Find API key
        api_key_record = db.query(APIKey).filter(APIKey.key == api_key).first()
        
        if not api_key_record:
            return None
        
        # Check if valid
        if not api_key_record.is_valid():
            return None
        
        # Update last used
        api_key_record.last_used = datetime.utcnow()
        db.commit()
        
        return api_key_record
    
    @staticmethod
    def check_rate_limit(db: Session, api_key_id: int) -> tuple[bool, str]:
        """
        Check rate limit for an API key
        
        Returns:
            Tuple of (allowed, reason)
        """
        # Get rate limit tracker
        tracker = db.query(RateLimitTracker).filter(
            RateLimitTracker.api_key_id == api_key_id
        ).first()
        
        if not tracker:
            return False, "Rate limit tracker not found"
        
        # Reset counters if needed
        if tracker.should_reset_minute():
            tracker.reset_minute()
        
        if tracker.should_reset_day():
            tracker.reset_day()
        
        db.commit()
        
        # Get API key limits
        api_key = db.query(APIKey).filter(APIKey.id == api_key_id).first()
        
        if not api_key:
            return False, "API key not found"
        
        # Check minute limit
        if tracker.requests_per_minute >= api_key.rate_limit_per_minute:
            return False, f"Rate limit exceeded: {api_key.rate_limit_per_minute} requests per minute"
        
        # Check day limit
        if tracker.requests_per_day >= api_key.rate_limit_per_day:
            return False, f"Rate limit exceeded: {api_key.rate_limit_per_day} requests per day"
        
        return True, "OK"
    
    @staticmethod
    def increment_rate_limit(db: Session, api_key_id: int, tokens_used: int = 0):
        """
        Increment rate limit counters
        """
        tracker = db.query(RateLimitTracker).filter(
            RateLimitTracker.api_key_id == api_key_id
        ).first()
        
        if tracker:
            tracker.requests_per_minute += 1
            tracker.requests_per_day += 1
            tracker.tokens_per_day += tokens_used
            db.commit()
    
    @staticmethod
    def create_subscription(
        db: Session,
        user_id: int,
        tier: str = "free",
        duration_days: Optional[int] = None
    ) -> Subscription:
        """
        Create a subscription for a user
        """
        # Define tier limits and pricing
        tier_config = {
            "free": {
                "monthly_tokens": 100000,
                "monthly_requests": 1000,
                "price": 0.0
            },
            "basic": {
                "monthly_tokens": 1000000,
                "monthly_requests": 10000,
                "price": 9.99
            },
            "premium": {
                "monthly_tokens": 10000000,
                "monthly_requests": 100000,
                "price": 49.99
            },
            "enterprise": {
                "monthly_tokens": None,  # Unlimited
                "monthly_requests": None,  # Unlimited
                "price": 299.99
            }
        }
        
        config = tier_config.get(tier, tier_config["free"])
        
        # Calculate expiration
        expires_at = None
        if duration_days:
            expires_at = datetime.utcnow() + timedelta(days=duration_days)
        elif tier != "free":
            expires_at = datetime.utcnow() + timedelta(days=30)
        
        # Create subscription
        subscription = Subscription(
            user_id=user_id,
            tier=tier,
            status="active",
            monthly_token_limit=config["monthly_tokens"],
            monthly_request_limit=config["monthly_requests"],
            price_per_month=config["price"],
            currency="USD",
            expires_at=expires_at
        )
        
        db.add(subscription)
        db.commit()
        db.refresh(subscription)
        
        return subscription
    
    @staticmethod
    def check_subscription(db: Session, user_id: int) -> tuple[bool, Optional[Subscription]]:
        """
        Check if user has active subscription
        """
        subscription = db.query(Subscription).filter(
            Subscription.user_id == user_id,
            Subscription.status == "active"
        ).order_by(Subscription.created_at.desc()).first()
        
        if not subscription:
            return False, None
        
        if not subscription.is_active():
            return False, subscription
        
        return True, subscription
    
    @staticmethod
    def log_usage(
        db: Session,
        user_id: int,
        api_key_id: int,
        endpoint: str,
        method: str,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        tokens_used: int = 0,
        generation_time: Optional[float] = None,
        status_code: int = 200,
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> UsageLog:
        """
        Log API usage
        """
        usage_log = UsageLog(
            user_id=user_id,
            api_key_id=api_key_id,
            endpoint=endpoint,
            method=method,
            prompt=prompt,
            response=response,
            tokens_used=tokens_used,
            generation_time=generation_time,
            status_code=status_code,
            error_message=error_message,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        db.add(usage_log)
        db.commit()
        db.refresh(usage_log)
        
        return usage_log
    
    @staticmethod
    def get_user_stats(db: Session, user_id: int) -> dict:
        """
        Get user usage statistics
        """
        # Get total usage
        total_requests = db.query(UsageLog).filter(
            UsageLog.user_id == user_id
        ).count()
        
        total_tokens = db.query(UsageLog).filter(
            UsageLog.user_id == user_id
        ).with_entities(UsageLog.tokens_used).all()
        
        total_tokens_sum = sum(t[0] for t in total_tokens if t[0])
        
        # Get current month usage
        first_day_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        monthly_requests = db.query(UsageLog).filter(
            UsageLog.user_id == user_id,
            UsageLog.created_at >= first_day_of_month
        ).count()
        
        monthly_tokens = db.query(UsageLog).filter(
            UsageLog.user_id == user_id,
            UsageLog.created_at >= first_day_of_month
        ).with_entities(UsageLog.tokens_used).all()
        
        monthly_tokens_sum = sum(t[0] for t in monthly_tokens if t[0])
        
        # Get subscription
        has_subscription, subscription = AuthManager.check_subscription(db, user_id)
        
        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens_sum,
            "monthly_requests": monthly_requests,
            "monthly_tokens": monthly_tokens_sum,
            "subscription": {
                "active": has_subscription,
                "tier": subscription.tier if subscription else None,
                "monthly_token_limit": subscription.monthly_token_limit if subscription else None,
                "monthly_request_limit": subscription.monthly_request_limit if subscription else None
            }
        }

# Helper functions for FastAPI dependencies
def get_current_user(db: Session, api_key: str) -> Optional[User]:
    """
    Get current user from API key
    """
    api_key_record = AuthManager.validate_api_key(db, api_key)
    
    if not api_key_record:
        return None
    
    user = db.query(User).filter(User.id == api_key_record.user_id).first()
    
    return user

def require_active_subscription(db: Session, user_id: int) -> bool:
    """
    Check if user has active subscription
    """
    has_subscription, _ = AuthManager.check_subscription(db, user_id)
    return has_subscription
