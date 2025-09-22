#!/usr/bin/env python3
"""
Test the API interface to identify any issues.
"""

import sys
import os
import logging
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_api_interface():
    """Test the API interface comprehensively."""
    
    print("🔍 Testing API Interface")
    print("=" * 50)
    
    # Test 1: Direct API connectivity
    print("\n1️⃣ Testing direct API connectivity...")
    try:
        import requests
        
        # Test basic CoinGecko API
        url = "https://api.coingecko.com/api/v3/ping"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print("✅ CoinGecko API is accessible")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ CoinGecko API error: {response.status_code}")
            return
            
    except Exception as e:
        print(f"❌ API connectivity test failed: {e}")
        return
    
    # Test 2: Test our hardened fetcher
    print("\n2️⃣ Testing hardened fetcher...")
    try:
        from data.fetch_coingecko import fetch_series, COINGECKO_MAP
        
        print(f"   Available symbols: {list(COINGECKO_MAP.keys())}")
        
        # Test with USDT (should always work)
        print("   Testing USDT fetch...")
        df = fetch_series('USDT', vs_currency='usd', days=7)  # Small test
        
        if not df.empty:
            print(f"✅ Hardened fetcher works!")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"   Sample data:")
            print(df.head(3).to_string(index=False))
        else:
            print("❌ Hardened fetcher returned empty data")
            return
            
    except Exception as e:
        print(f"❌ Hardened fetcher test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Test configuration mapping
    print("\n3️⃣ Testing configuration mapping...")
    try:
        from utils.config import load_config
        
        config = load_config('configs/config.yaml')
        stablecoins = config.get('stablecoins', {})
        tickers = stablecoins.get('tickers', [])
        coingecko_ids = stablecoins.get('coingecko_ids', {})
        
        print(f"   Configured tickers: {tickers}")
        print(f"   CoinGecko IDs: {coingecko_ids}")
        
        # Check if all tickers have mappings
        missing_mappings = []
        for ticker in tickers:
            if ticker not in coingecko_ids:
                missing_mappings.append(ticker)
        
        if missing_mappings:
            print(f"⚠️ Missing CoinGecko mappings: {missing_mappings}")
        else:
            print("✅ All tickers have CoinGecko mappings")
            
        # Check if mappings exist in our fetcher
        fetcher_missing = []
        for ticker, coin_id in coingecko_ids.items():
            if ticker not in COINGECKO_MAP:
                fetcher_missing.append(ticker)
        
        if fetcher_missing:
            print(f"❌ Missing in fetcher MAP: {fetcher_missing}")
            print(f"   Current MAP: {COINGECKO_MAP}")
        else:
            print("✅ All configured tickers are in fetcher MAP")
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return
    
    # Test 4: Test pipeline integration
    print("\n4️⃣ Testing pipeline integration...")
    try:
        from pipelines.run_ingest import fetch_stablecoin_data
        
        # Test with small date range
        start_date = '2024-01-01'
        end_date = '2024-01-07'
        
        print(f"   Testing with date range: {start_date} to {end_date}")
        
        stablecoin_data = fetch_stablecoin_data(config, start_date, end_date)
        
        if stablecoin_data:
            print("✅ Pipeline integration works!")
            for key, df in stablecoin_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    print(f"   {key}: {df.shape}")
                    print(f"   Columns: {list(df.columns)}")
                    print(f"   Date range: {df.index.min()} to {df.index.max()}")
                else:
                    print(f"   {key}: Empty or not DataFrame")
        else:
            print("❌ Pipeline integration returned no data")
            
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 API Interface test completed!")
    
    # Summary
    print("\n📋 Summary:")
    print("If all tests pass, the API interface should work correctly.")
    print("If any test fails, that's where the issue is.")
    print("\nNext steps:")
    print("1. Run: python -m src.data.fetch_coingecko")
    print("2. Run: run_all_windows.bat")

if __name__ == "__main__":
    test_api_interface()

