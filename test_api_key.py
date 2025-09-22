#!/usr/bin/env python3
"""
Simple test to verify DeepSeek API key works.
"""

import os
import requests
import json
from dotenv import load_dotenv

def test_deepseek_api():
    """Test DeepSeek API with the provided key."""
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('DEEPSEEK_API_KEY', '')
    
    print("🧪 Testing DeepSeek API Key")
    print("=" * 40)
    print(f"API Key: {api_key[:10]}...")
    
    # Test API call
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Please respond with 'API test successful' if you can read this."}
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }
    
    try:
        print("\n📡 Making API call...")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content']
            print(f"✅ API Response: {message}")
            print("🎉 DeepSeek API key is working!")
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_deepseek_api()
