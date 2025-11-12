#!/usr/bin/env python3
"""
User Management CLI Tool for dLNk GPT
"""

import argparse
from sqlalchemy.orm import Session
from app.database import SessionLocal, User, APIKey, Subscription
from app.auth import AuthManager
from datetime import datetime
import sys

def create_user_command(args):
    """
    Create a new user
    """
    db = SessionLocal()
    
    try:
        user = AuthManager.create_user(
            db=db,
            username=args.username,
            email=args.email,
            password=args.password,
            full_name=args.full_name
        )
        
        print(f"✓ User created successfully!")
        print(f"  ID: {user.id}")
        print(f"  Username: {user.username}")
        print(f"  Email: {user.email}")
        print(f"  Full Name: {user.full_name}")
        
        # Create API key if requested
        if args.create_api_key:
            api_key = AuthManager.create_api_key(
                db=db,
                user_id=user.id,
                name=f"{user.username}'s API Key"
            )
            print(f"\n✓ API Key created:")
            print(f"  {api_key.key}")
            print(f"  (Save this key - it won't be shown again)")
        
    except ValueError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    finally:
        db.close()

def list_users_command(args):
    """
    List all users
    """
    db = SessionLocal()
    
    try:
        users = db.query(User).all()
        
        if not users:
            print("No users found.")
            return
        
        print(f"\nTotal users: {len(users)}\n")
        print(f"{'ID':<5} {'Username':<20} {'Email':<30} {'Active':<8} {'Admin':<8} {'Created':<20}")
        print("-" * 100)
        
        for user in users:
            print(f"{user.id:<5} {user.username:<20} {user.email:<30} "
                  f"{'Yes' if user.is_active else 'No':<8} "
                  f"{'Yes' if user.is_admin else 'No':<8} "
                  f"{user.created_at.strftime('%Y-%m-%d %H:%M'):<20}")
        
    finally:
        db.close()

def create_api_key_command(args):
    """
    Create API key for a user
    """
    db = SessionLocal()
    
    try:
        # Find user
        if args.user_id:
            user = db.query(User).filter(User.id == args.user_id).first()
        elif args.username:
            user = db.query(User).filter(User.username == args.username).first()
        else:
            print("✗ Error: Must specify --user-id or --username")
            sys.exit(1)
        
        if not user:
            print("✗ Error: User not found")
            sys.exit(1)
        
        # Create API key
        api_key = AuthManager.create_api_key(
            db=db,
            user_id=user.id,
            name=args.name,
            expires_in_days=args.expires_in_days,
            rate_limit_per_minute=args.rate_limit_minute,
            rate_limit_per_day=args.rate_limit_day
        )
        
        print(f"✓ API Key created successfully!")
        print(f"  User: {user.username}")
        print(f"  Key: {api_key.key}")
        print(f"  Name: {api_key.name}")
        print(f"  Rate Limit: {api_key.rate_limit_per_minute}/min, {api_key.rate_limit_per_day}/day")
        if api_key.expires_at:
            print(f"  Expires: {api_key.expires_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"\n⚠️  Save this key - it won't be shown again!")
        
    finally:
        db.close()

def list_api_keys_command(args):
    """
    List API keys
    """
    db = SessionLocal()
    
    try:
        query = db.query(APIKey)
        
        if args.user_id:
            query = query.filter(APIKey.user_id == args.user_id)
        elif args.username:
            user = db.query(User).filter(User.username == args.username).first()
            if user:
                query = query.filter(APIKey.user_id == user.id)
        
        api_keys = query.all()
        
        if not api_keys:
            print("No API keys found.")
            return
        
        print(f"\nTotal API keys: {len(api_keys)}\n")
        print(f"{'ID':<5} {'User ID':<8} {'Name':<25} {'Active':<8} {'Created':<20}")
        print("-" * 80)
        
        for key in api_keys:
            print(f"{key.id:<5} {key.user_id:<8} {key.name:<25} "
                  f"{'Yes' if key.is_active else 'No':<8} "
                  f"{key.created_at.strftime('%Y-%m-%d %H:%M'):<20}")
        
    finally:
        db.close()

def update_subscription_command(args):
    """
    Update user subscription
    """
    db = SessionLocal()
    
    try:
        # Find user
        if args.user_id:
            user = db.query(User).filter(User.id == args.user_id).first()
        elif args.username:
            user = db.query(User).filter(User.username == args.username).first()
        else:
            print("✗ Error: Must specify --user-id or --username")
            sys.exit(1)
        
        if not user:
            print("✗ Error: User not found")
            sys.exit(1)
        
        # Deactivate old subscriptions
        old_subs = db.query(Subscription).filter(
            Subscription.user_id == user.id,
            Subscription.status == "active"
        ).all()
        
        for sub in old_subs:
            sub.status = "cancelled"
        
        # Create new subscription
        subscription = AuthManager.create_subscription(
            db=db,
            user_id=user.id,
            tier=args.tier,
            duration_days=args.duration_days
        )
        
        print(f"✓ Subscription updated successfully!")
        print(f"  User: {user.username}")
        print(f"  Tier: {subscription.tier}")
        print(f"  Monthly Token Limit: {subscription.monthly_token_limit or 'Unlimited'}")
        print(f"  Monthly Request Limit: {subscription.monthly_request_limit or 'Unlimited'}")
        print(f"  Price: ${subscription.price_per_month}/month")
        if subscription.expires_at:
            print(f"  Expires: {subscription.expires_at.strftime('%Y-%m-%d %H:%M')}")
        
    finally:
        db.close()

def user_stats_command(args):
    """
    Show user statistics
    """
    db = SessionLocal()
    
    try:
        # Find user
        if args.user_id:
            user = db.query(User).filter(User.id == args.user_id).first()
        elif args.username:
            user = db.query(User).filter(User.username == args.username).first()
        else:
            print("✗ Error: Must specify --user-id or --username")
            sys.exit(1)
        
        if not user:
            print("✗ Error: User not found")
            sys.exit(1)
        
        # Get stats
        stats = AuthManager.get_user_stats(db, user.id)
        
        print(f"\n{'=' * 70}")
        print(f"User Statistics: {user.username}")
        print(f"{'=' * 70}")
        print(f"\nAccount Information:")
        print(f"  Email: {user.email}")
        print(f"  Full Name: {user.full_name or 'N/A'}")
        print(f"  Active: {'Yes' if user.is_active else 'No'}")
        print(f"  Created: {user.created_at.strftime('%Y-%m-%d %H:%M')}")
        if user.last_login:
            print(f"  Last Login: {user.last_login.strftime('%Y-%m-%d %H:%M')}")
        
        print(f"\nUsage Statistics:")
        print(f"  Total Requests: {stats['total_requests']:,}")
        print(f"  Total Tokens: {stats['total_tokens']:,}")
        print(f"  Monthly Requests: {stats['monthly_requests']:,}")
        print(f"  Monthly Tokens: {stats['monthly_tokens']:,}")
        
        print(f"\nSubscription:")
        print(f"  Active: {'Yes' if stats['subscription']['active'] else 'No'}")
        print(f"  Tier: {stats['subscription']['tier'] or 'None'}")
        if stats['subscription']['monthly_token_limit']:
            print(f"  Token Limit: {stats['subscription']['monthly_token_limit']:,}")
        else:
            print(f"  Token Limit: Unlimited")
        if stats['subscription']['monthly_request_limit']:
            print(f"  Request Limit: {stats['subscription']['monthly_request_limit']:,}")
        else:
            print(f"  Request Limit: Unlimited")
        
        print(f"{'=' * 70}\n")
        
    finally:
        db.close()

def main():
    """
    Main CLI function
    """
    parser = argparse.ArgumentParser(
        description="dLNk GPT User Management Tool"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Create user
    create_user_parser = subparsers.add_parser("create-user", help="Create a new user")
    create_user_parser.add_argument("--username", required=True, help="Username")
    create_user_parser.add_argument("--email", required=True, help="Email address")
    create_user_parser.add_argument("--password", required=True, help="Password")
    create_user_parser.add_argument("--full-name", help="Full name")
    create_user_parser.add_argument("--create-api-key", action="store_true", help="Create API key")
    
    # List users
    list_users_parser = subparsers.add_parser("list-users", help="List all users")
    
    # Create API key
    create_key_parser = subparsers.add_parser("create-api-key", help="Create API key")
    create_key_parser.add_argument("--user-id", type=int, help="User ID")
    create_key_parser.add_argument("--username", help="Username")
    create_key_parser.add_argument("--name", help="API key name")
    create_key_parser.add_argument("--expires-in-days", type=int, help="Expiration in days")
    create_key_parser.add_argument("--rate-limit-minute", type=int, default=60, help="Rate limit per minute")
    create_key_parser.add_argument("--rate-limit-day", type=int, default=10000, help="Rate limit per day")
    
    # List API keys
    list_keys_parser = subparsers.add_parser("list-api-keys", help="List API keys")
    list_keys_parser.add_argument("--user-id", type=int, help="Filter by user ID")
    list_keys_parser.add_argument("--username", help="Filter by username")
    
    # Update subscription
    update_sub_parser = subparsers.add_parser("update-subscription", help="Update user subscription")
    update_sub_parser.add_argument("--user-id", type=int, help="User ID")
    update_sub_parser.add_argument("--username", help="Username")
    update_sub_parser.add_argument("--tier", required=True, choices=["free", "basic", "premium", "enterprise"], help="Subscription tier")
    update_sub_parser.add_argument("--duration-days", type=int, help="Duration in days")
    
    # User stats
    stats_parser = subparsers.add_parser("user-stats", help="Show user statistics")
    stats_parser.add_argument("--user-id", type=int, help="User ID")
    stats_parser.add_argument("--username", help="Username")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == "create-user":
        create_user_command(args)
    elif args.command == "list-users":
        list_users_command(args)
    elif args.command == "create-api-key":
        create_api_key_command(args)
    elif args.command == "list-api-keys":
        list_api_keys_command(args)
    elif args.command == "update-subscription":
        update_subscription_command(args)
    elif args.command == "user-stats":
        user_stats_command(args)

if __name__ == "__main__":
    main()
