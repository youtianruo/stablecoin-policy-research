#!/usr/bin/env python3
"""
Simple test for the multi-provider fetcher.
"""

import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_fetcher():
    """Test the multi-provider fetcher."""
    
    print("🧪 Testing Multi-Provider Fetcher")
    print("=" * 40)
    
    try:
        from data.fetch_coingecko import fetch_series
        print("✅ Import successful")
        
        # Test with USDT
        print("\n📊 Testing USDT fetch...")
        df = fetch_series("USDT", vs_currency="usd", days=7)
        
        if not df.empty:
            print(f"✅ Fetch successful!")
            print(f"   Shape: {df.shape}")
            print(f"   Provider: {df['provider'].iloc[0]}")
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print("   Sample data:")
            print(df.head(3)[['timestamp', 'symbol', 'price', 'provider']].to_string(index=False))
        else:
            print("❌ Empty data returned")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    test_fetcher()
