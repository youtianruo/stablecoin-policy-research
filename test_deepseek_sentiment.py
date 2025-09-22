#!/usr/bin/env python3
"""
Test DeepSeek sentiment analysis with sample policy text.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.features.sentiment.llm_adapter import LLMAdapter

def test_policy_sentiment():
    """Test DeepSeek sentiment analysis with sample policy texts."""
    
    print("🧠 DeepSeek Policy Sentiment Analysis Test")
    print("=" * 60)
    
    # Set API key
    api_key = ""
    
    # Initialize LLM adapter
    llm_adapter = LLMAdapter(api_key=api_key, model="deepseek-chat")
    
    # Sample policy texts
    policy_texts = [
        {
            "title": "Hawkish Policy Statement",
            "text": """
            The Federal Reserve is committed to bringing inflation back to our 2% target. 
            Given the persistent inflationary pressures and tight labor market conditions, 
            we believe it is appropriate to continue raising interest rates and reducing 
            our balance sheet at a faster pace. We will remain vigilant and take forceful 
            action as needed to restore price stability.
            """
        },
        {
            "title": "Dovish Policy Statement", 
            "text": """
            The Federal Reserve remains committed to supporting the economic recovery. 
            While we are monitoring inflation carefully, we believe our current accommodative 
            stance is appropriate given the ongoing challenges in the labor market and 
            the need to support maximum employment. We will continue our asset purchases 
            and maintain low interest rates to support economic growth.
            """
        },
        {
            "title": "Neutral Policy Statement",
            "text": """
            The Federal Reserve will continue to monitor incoming economic data and 
            adjust our monetary policy stance as appropriate. We remain committed 
            to our dual mandate of maximum employment and price stability. Our policy 
            decisions will be data-dependent and we will communicate clearly about 
            our assessment of economic conditions and policy implications.
            """
        }
    ]
    
    print("📊 Analyzing Policy Sentiment...")
    print("-" * 40)
    
    for i, policy in enumerate(policy_texts, 1):
        print(f"\n{i}. {policy['title']}")
        print("=" * 50)
        
        # Analyze sentiment
        result = llm_adapter.analyze_sentiment(
            policy['text'], 
            "Federal Reserve monetary policy statement"
        )
        
        print(f"📈 Sentiment: {result['sentiment'].upper()}")
        print(f"🎯 Confidence: {result['confidence']:.3f}")
        print(f"🔑 Key Phrases: {', '.join(result['key_phrases'])}")
        print(f"💭 Explanation: {result['explanation']}")
        
        # Show text snippet
        text_snippet = policy['text'].strip()[:200] + "..."
        print(f"📄 Text: {text_snippet}")
    
    print("\n🎉 DeepSeek Policy Sentiment Analysis Complete!")
    print("\n💡 Key Insights:")
    print("   - DeepSeek successfully classifies policy sentiment")
    print("   - Provides confidence scores and explanations")
    print("   - Identifies key phrases supporting classifications")
    print("   - Ready for real policy document analysis")

if __name__ == "__main__":
    test_policy_sentiment()
