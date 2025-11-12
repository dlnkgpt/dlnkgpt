#!/usr/bin/env python3
"""
Test script for dLNk GPT API
"""

import requests
import json
import sys
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "demo_key_123"

def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_result(success: bool, message: str):
    """Print test result"""
    status = "✓" if success else "✗"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status} {message}{reset}")

def test_health_check() -> bool:
    """Test health check endpoint"""
    print_header("Test 1: Health Check")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        data = response.json()
        
        if response.status_code == 200 and data.get("status") == "healthy":
            print_result(True, f"Health check passed: {json.dumps(data, indent=2)}")
            return True
        else:
            print_result(False, f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_result(False, f"Health check error: {e}")
        return False

def test_status_endpoint() -> bool:
    """Test status endpoint"""
    print_header("Test 2: Status Endpoint")
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        data = response.json()
        
        if response.status_code == 200:
            print_result(True, f"Status endpoint passed:")
            print(json.dumps(data, indent=2))
            return True
        else:
            print_result(False, f"Status endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print_result(False, f"Status endpoint error: {e}")
        return False

def test_model_info() -> bool:
    """Test model info endpoint"""
    print_header("Test 3: Model Info")
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        data = response.json()
        
        if response.status_code == 200:
            print_result(True, f"Model info retrieved:")
            print(json.dumps(data, indent=2))
            return True
        else:
            print_result(False, f"Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print_result(False, f"Model info error: {e}")
        return False

def test_chat_valid_key() -> bool:
    """Test chat endpoint with valid API key"""
    print_header("Test 4: Chat with Valid API Key")
    try:
        payload = {
            "api_key": API_KEY,
            "prompt": "Write a hello world program in Python",
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print_result(True, "Chat request successful:")
            print(f"Response: {data.get('response', '')[:200]}...")
            print(f"Tokens used: {data.get('tokens_used', 0)}")
            return True
        else:
            print_result(False, f"Chat request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print_result(False, f"Chat request error: {e}")
        return False

def test_chat_invalid_key() -> bool:
    """Test chat endpoint with invalid API key"""
    print_header("Test 5: Chat with Invalid API Key")
    try:
        payload = {
            "api_key": "invalid_key_xyz",
            "prompt": "Test prompt",
            "max_tokens": 100
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 401:
            print_result(True, "Invalid API key correctly rejected (401)")
            return True
        else:
            print_result(False, f"Expected 401, got {response.status_code}")
            return False
    except Exception as e:
        print_result(False, f"Invalid key test error: {e}")
        return False

def test_chat_missing_fields() -> bool:
    """Test chat endpoint with missing required fields"""
    print_header("Test 6: Chat with Missing Fields")
    try:
        payload = {
            "api_key": API_KEY
            # Missing 'prompt' field
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 422:
            print_result(True, "Missing fields correctly rejected (422)")
            return True
        else:
            print_result(False, f"Expected 422, got {response.status_code}")
            return False
    except Exception as e:
        print_result(False, f"Missing fields test error: {e}")
        return False

def test_api_docs() -> bool:
    """Test API documentation endpoint"""
    print_header("Test 7: API Documentation")
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
        
        if response.status_code == 200:
            print_result(True, "API documentation accessible at /docs")
            return True
        else:
            print_result(False, f"API docs failed: {response.status_code}")
            return False
    except Exception as e:
        print_result(False, f"API docs error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("  dLNk GPT API Test Suite")
    print("=" * 70)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"API Key: {API_KEY}")
    
    tests = [
        ("Health Check", test_health_check),
        ("Status Endpoint", test_status_endpoint),
        ("Model Info", test_model_info),
        ("Chat (Valid Key)", test_chat_valid_key),
        ("Chat (Invalid Key)", test_chat_invalid_key),
        ("Chat (Missing Fields)", test_chat_missing_fields),
        ("API Documentation", test_api_docs),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_result(False, f"{name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print_header("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        print_result(result, name)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    try:
        exit_code = run_all_tests()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
