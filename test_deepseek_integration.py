#!/usr/bin/env python3
"""
Test DeepSeek API integration for policy sentiment analysis.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.features.sentiment.llm_adapter import LLMAdapter
from src.utils.config import load_config

def test_deepseek_integration():
    """Test DeepSeek API integration."""
    
    print("🧪 Testing DeepSeek API Integration")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("❌ DEEPSEEK_API_KEY not found in environment variables")
        print("💡 To test DeepSeek integration:")
        print("   1. Get API key from: https://platform.deepseek.com/")
        print("   2. Create .env file with: DEEPSEEK_API_KEY=your_key_here")
        print("   3. Run this test again")
        return False
    
    print(f"✅ Found DeepSeek API key: {api_key[:10]}...")
    
    try:
        # Initialize LLM adapter
        print("\n🔧 Initializing DeepSeek LLM adapter...")
        llm_adapter = LLMAdapter(api_key=api_key, model="deepseek-chat")
        print("✅ LLM adapter initialized successfully")
        
        # Test sentiment analysis
        print("\n📊 Testing sentiment analysis...")
        test_text = """
        The Federal Reserve remains committed to achieving maximum employment and price stability. 
        Given the current economic conditions and inflationary pressures, we believe it is appropriate 
        to maintain our current accommodative stance while monitoring incoming data carefully.
        """
        
        result = llm_adapter.analyze_sentiment(test_text, "Federal Reserve policy statement")
        
        print("✅ Sentiment analysis completed!")
        print(f"   Sentiment: {result['sentiment']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Key phrases: {result['key_phrases']}")
        print(f"   Explanation: {result['explanation']}")
        
        # Test batch analysis
        print("\n📋 Testing batch analysis...")
        test_texts = [
            "The Fed is considering raising interest rates to combat inflation.",
            "We will continue our accommodative monetary policy to support economic recovery.",
            "The committee will monitor economic data before making any policy decisions."
        ]
        
        batch_results = llm_adapter.analyze_batch(test_texts)
        
        print("✅ Batch analysis completed!")
        for i, result in enumerate(batch_results):
            print(f"   Text {i+1}: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        
        # Test configuration loading
        print("\n⚙️ Testing configuration integration...")
        config = load_config("configs/config.yaml")
        sentiment_config = config.get('sentiment', {})
        print(f"   Model: {sentiment_config.get('model', 'Not set')}")
        print(f"   Categories: {sentiment_config.get('categories', [])}")
        
        print("\n🎉 DeepSeek integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing DeepSeek integration: {e}")
        print("\n🔍 Troubleshooting:")
        print("   1. Check your API key is valid")
        print("   2. Ensure you have internet connection")
        print("   3. Verify DeepSeek API is accessible")
        print("   4. Check API rate limits and quotas")
        return False

def test_configuration():
    """Test configuration loading with DeepSeek."""
    
    print("\n⚙️ Testing Configuration Integration")
    print("=" * 40)
    
    try:
        config = load_config("configs/config.yaml")
        
        # Check sentiment configuration
        sentiment_config = config.get('sentiment', {})
        print(f"✅ Sentiment model: {sentiment_config.get('model')}")
        print(f"✅ Sentiment categories: {sentiment_config.get('categories')}")
        
        # Check API keys section
        api_keys = config.get('api_keys', {})
        print(f"✅ API keys configured: {list(api_keys.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Main test function."""
    
    print("🚀 DeepSeek Integration Test Suite")
    print("=" * 60)
    
    # Test configuration
    config_ok = test_configuration()
    
    # Test API integration
    api_ok = test_deepseek_integration()
    
    print("\n📊 Test Results Summary")
    print("=" * 30)
    print(f"Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"API Integration: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if config_ok and api_ok:
        print("\n🎉 All tests passed! DeepSeek integration is ready.")
        print("\n💡 Next steps:")
        print("   1. Add your DeepSeek API key to .env file")
        print("   2. Run the full pipeline: run_all_windows.bat")
        print("   3. Check data/processed/ for sentiment analysis results")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
        print("\n🔧 Setup instructions:")
        print("   1. Get DeepSeek API key: https://platform.deepseek.com/")
        print("   2. Create .env file with: DEEPSEEK_API_KEY=your_key_here")
        print("   3. Run: py test_deepseek_integration.py")

if __name__ == "__main__":
    main()
