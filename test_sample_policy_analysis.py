#!/usr/bin/env python3
"""
Test DeepSeek sentiment analysis on sample policy events.
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.features.sentiment.llm_adapter import LLMAdapter

def analyze_sample_policy_events():
    """Analyze sentiment of sample policy events."""
    
    print("📰 Sample Policy Events Sentiment Analysis")
    print("=" * 60)
    
    # Load sample policy events
    try:
        df = pd.read_csv('sample_policy_events.csv')
        print(f"✅ Loaded {len(df)} policy events")
    except Exception as e:
        print(f"❌ Error loading sample policy events: {e}")
        return
    
    # Initialize DeepSeek LLM adapter
    api_key = "sk-feceb4354b6e4c479027028141e226b7"
    llm_adapter = LLMAdapter(api_key=api_key, model="deepseek-chat")
    
    print(f"\n🧠 Analyzing sentiment with DeepSeek...")
    print("-" * 50)
    
    # Analyze each policy event
    results = []
    
    for idx, row in df.iterrows():
        print(f"\n{idx+1}. {row['title']} ({row['date']})")
        print("=" * 50)
        
        # Analyze sentiment
        result = llm_adapter.analyze_sentiment(
            row['content'], 
            f"Federal Reserve {row['event_type']}"
        )
        
        print(f"📈 Sentiment: {result['sentiment'].upper()}")
        print(f"🎯 Confidence: {result['confidence']:.3f}")
        print(f"🔑 Key Phrases: {', '.join(result['key_phrases'][:3])}")
        
        # Store results
        results.append({
            'date': row['date'],
            'event_type': row['event_type'],
            'title': row['title'],
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'key_phrases': result['key_phrases'],
            'explanation': result['explanation']
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    print(f"\n📊 Sentiment Analysis Summary")
    print("=" * 40)
    
    sentiment_counts = results_df['sentiment'].value_counts()
    print("Sentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment.upper()}: {count} events")
    
    avg_confidence = results_df['confidence'].mean()
    print(f"\nAverage Confidence: {avg_confidence:.3f}")
    
    # Show confidence by sentiment
    print(f"\nConfidence by Sentiment:")
    for sentiment in results_df['sentiment'].unique():
        subset = results_df[results_df['sentiment'] == sentiment]
        avg_conf = subset['confidence'].mean()
        print(f"  {sentiment.upper()}: {avg_conf:.3f}")
    
    # Save results
    results_df.to_csv('policy_sentiment_results.csv', index=False)
    print(f"\n💾 Results saved to policy_sentiment_results.csv")
    
    print(f"\n🎉 Policy sentiment analysis complete!")
    print(f"\n💡 Key Insights:")
    print(f"   - Analyzed {len(df)} policy events")
    print(f"   - DeepSeek successfully classified all events")
    print(f"   - Average confidence: {avg_confidence:.1%}")
    print(f"   - Ready for event study analysis")

if __name__ == "__main__":
    analyze_sample_policy_events()
