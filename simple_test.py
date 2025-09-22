#!/usr/bin/env python3
"""
Simple test for multi-provider fetcher.
"""

print("🧪 Simple Multi-Provider Test")
print("=" * 30)

try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    print("✅ Path setup successful")
    
    from data.fetch_coingecko import fetch_series
    print("✅ Import successful")
    
    # Test with USDT
    print("📊 Testing USDT fetch...")
    df = fetch_series("USDT", vs_currency="usd", days=7)
    
    if not df.empty:
        print(f"✅ Fetch successful! Shape: {df.shape}")
        print(f"Provider: {df['provider'].iloc[0]}")
        print("Sample data:")
        print(df.head(2))
    else:
        print("❌ Empty data returned")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n🏁 Test completed!")
