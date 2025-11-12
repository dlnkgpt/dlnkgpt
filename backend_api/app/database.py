"""
Database configuration and models for dLNk GPT
Using SQLAlchemy with PostgreSQL/SQLite support
"""

from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timedelta
import os
from typing import Optional

# Database URL from environment or default to SQLite
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///./dlnkgpt.db"
)

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False
)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Database dependency
def get_db():
    """
    Get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Models
class User(Base):
    """
    User model
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    subscriptions = relationship("Subscription", back_populates="user", cascade="all, delete-orphan")
    usage_logs = relationship("UsageLog", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

class APIKey(Base):
    """
    API Key model
    """
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    key = Column(String(255), unique=True, index=True, nullable=False)
    key_hash = Column(String(255), nullable=False)
    name = Column(String(100), nullable=True)
    
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    last_used = Column(DateTime, nullable=True)
    
    # Rate limiting
    rate_limit_per_minute = Column(Integer, default=60)
    rate_limit_per_day = Column(Integer, default=10000)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    usage_logs = relationship("UsageLog", back_populates="api_key", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<APIKey(id={self.id}, user_id={self.user_id}, name='{self.name}')>"
    
    def is_valid(self) -> bool:
        """
        Check if API key is valid
        """
        if not self.is_active:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True

class Subscription(Base):
    """
    Subscription model
    """
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    tier = Column(String(50), nullable=False)  # free, basic, premium, enterprise
    status = Column(String(50), default="active")  # active, cancelled, expired, suspended
    
    # Limits
    monthly_token_limit = Column(Integer, nullable=True)
    monthly_request_limit = Column(Integer, nullable=True)
    
    # Billing
    price_per_month = Column(Float, default=0.0)
    currency = Column(String(10), default="USD")
    
    # Dates
    started_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="subscriptions")
    
    def __repr__(self):
        return f"<Subscription(id={self.id}, user_id={self.user_id}, tier='{self.tier}', status='{self.status}')>"
    
    def is_active(self) -> bool:
        """
        Check if subscription is active
        """
        if self.status != "active":
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True
    
    def get_limits(self) -> dict:
        """
        Get subscription limits
        """
        return {
            "monthly_tokens": self.monthly_token_limit,
            "monthly_requests": self.monthly_request_limit
        }

class UsageLog(Base):
    """
    Usage log model for tracking API usage
    """
    __tablename__ = "usage_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=False)
    
    # Request details
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    
    # Prompt and response
    prompt = Column(Text, nullable=True)
    response = Column(Text, nullable=True)
    
    # Metrics
    tokens_used = Column(Integer, default=0)
    generation_time = Column(Float, nullable=True)
    
    # Status
    status_code = Column(Integer, default=200)
    error_message = Column(Text, nullable=True)
    
    # Metadata
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="usage_logs")
    api_key = relationship("APIKey", back_populates="usage_logs")
    
    def __repr__(self):
        return f"<UsageLog(id={self.id}, user_id={self.user_id}, endpoint='{self.endpoint}', tokens={self.tokens_used})>"

class RateLimitTracker(Base):
    """
    Rate limit tracking model
    """
    __tablename__ = "rate_limit_trackers"
    
    id = Column(Integer, primary_key=True, index=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=False)
    
    # Counters
    requests_per_minute = Column(Integer, default=0)
    requests_per_day = Column(Integer, default=0)
    tokens_per_day = Column(Integer, default=0)
    
    # Reset times
    minute_reset_at = Column(DateTime, nullable=False)
    day_reset_at = Column(DateTime, nullable=False)
    
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<RateLimitTracker(id={self.id}, api_key_id={self.api_key_id})>"
    
    def should_reset_minute(self) -> bool:
        """
        Check if minute counter should reset
        """
        return datetime.utcnow() >= self.minute_reset_at
    
    def should_reset_day(self) -> bool:
        """
        Check if day counter should reset
        """
        return datetime.utcnow() >= self.day_reset_at
    
    def reset_minute(self):
        """
        Reset minute counter
        """
        self.requests_per_minute = 0
        self.minute_reset_at = datetime.utcnow() + timedelta(minutes=1)
    
    def reset_day(self):
        """
        Reset day counter
        """
        self.requests_per_day = 0
        self.tokens_per_day = 0
        self.day_reset_at = datetime.utcnow() + timedelta(days=1)

def init_db():
    """
    Initialize database - create all tables
    """
    print("Initializing database...")
    Base.metadata.create_all(bind=engine)
    print("✓ Database initialized successfully")

def drop_db():
    """
    Drop all tables - use with caution!
    """
    print("⚠️  Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("✓ All tables dropped")

if __name__ == "__main__":
    # Initialize database when run directly
    init_db()
    
    # Print table information
    print("\nCreated tables:")
    for table in Base.metadata.tables.keys():
        print(f"  - {table}")
