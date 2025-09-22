#!/usr/bin/env python3
"""
Test the multi-provider data fetcher.
"""

import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_multi_provider():
    """Test the multi-provider fetcher."""
    
    print("🧪 Testing Multi-Provider Data Fetcher")
    print("=" * 50)
    
    try:
        from data.fetch_coingecko import fetch_series, fetch_coingecko, fetch_yahoo, fetch_coincap
        
        # Test symbols
        test_symbols = ["USDT", "USDC", "BTC"]
        
        for symbol in test_symbols:
            print(f"\n📊 Testing {symbol}...")
            
            try:
                # Test multi-provider fetch
                df = fetch_series(symbol, vs_currency="usd", days=7)
                
                if not df.empty:
                    print(f"✅ Multi-provider fetch successful!")
                    print(f"   Shape: {df.shape}")
                    print(f"   Provider: {df['provider'].iloc[0] if 'provider' in df.columns else 'Unknown'}")
                    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                    print(f"   Sample data:")
                    print(df.head(3)[['timestamp', 'symbol', 'price', 'provider']].to_string(index=False))
                else:
                    print(f"❌ Multi-provider fetch returned empty data")
                    
            except Exception as e:
                print(f"❌ Multi-provider fetch failed: {e}")
        
        # Test individual providers
        print(f"\n🔍 Testing individual providers for USDT...")
        
        providers = [
            ("CoinGecko", fetch_coingecko),
            ("Yahoo Finance", fetch_yahoo),
            ("CoinCap", fetch_coincap)
        ]
        
        for provider_name, fetch_func in providers:
            try:
                print(f"   Testing {provider_name}...")
                df = fetch_func("USDT", vs_currency="usd", days=7)
                
                if not df.empty:
                    print(f"   ✅ {provider_name}: {len(df)} data points")
                else:
                    print(f"   ⚠️ {provider_name}: Empty data")
                    
            except Exception as e:
                print(f"   ❌ {provider_name}: {e}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Multi-provider test completed!")
    print("\nNext steps:")
    print("1. Run: python -m src.data.fetch_coingecko")
    print("2. Run: run_all_windows.bat")

if __name__ == "__main__":
    test_multi_provider()
