#!/usr/bin/env python3
"""
Simple test script to verify data fetching works.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_simple_data_fetch():
    """Test basic data fetching without complex pipeline."""
    
    print("🧪 Testing Simple Data Fetch")
    print("=" * 40)
    
    try:
        # Test CoinGecko with a simple request
        print("\n1️⃣ Testing CoinGecko API...")
        
        import requests
        
        # Simple API test - get current Bitcoin price
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': 'bitcoin',
            'vs_currencies': 'usd'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ CoinGecko API accessible")
            print(f"   Bitcoin price: ${data['bitcoin']['usd']}")
        else:
            print(f"❌ CoinGecko API error: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ CoinGecko test failed: {e}")
    
    try:
        # Test FRED API
        print("\n2️⃣ Testing FRED API...")
        
        # Check if FRED key exists
        from utils.config import load_config
        config = load_config('configs/config.yaml')
        fred_key = config.get('api_keys', {}).get('fred')
        
        if fred_key:
            print(f"✅ FRED API key found")
            
            # Test simple FRED request
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': 'FEDFUNDS',
                'api_key': fred_key,
                'file_type': 'json',
                'limit': 1,
                'sort_order': 'desc'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'observations' in data and data['observations']:
                    latest_rate = data['observations'][0]['value']
                    print(f"✅ FRED API accessible")
                    print(f"   Latest Fed Funds Rate: {latest_rate}%")
                else:
                    print(f"⚠️ FRED API accessible but no data returned")
            else:
                print(f"❌ FRED API error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        else:
            print(f"⚠️ No FRED API key found in config")
            
    except Exception as e:
        print(f"❌ FRED test failed: {e}")
    
    try:
        # Test our data fetchers
        print("\n3️⃣ Testing our data fetchers...")
        
        from data.fetch_coingecko import CoinGeckoFetcher
        
        fetcher = CoinGeckoFetcher()
        
        # Test with a very small date range
        start_date = '2024-01-01'
        end_date = '2024-01-02'
        
        # Test with Bitcoin (should always work)
        prices = fetcher.get_stablecoin_prices(['bitcoin'], start_date, end_date)
        
        if not prices.empty:
            print(f"✅ Data fetcher working")
            print(f"   Data shape: {prices.shape}")
            print(f"   Date range: {prices.index.min()} to {prices.index.max()}")
        else:
            print(f"❌ Data fetcher returned empty data")
            
    except Exception as e:
        print(f"❌ Data fetcher test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Simple test completed!")
    print("\nIf all tests pass, the issue might be:")
    print("1. Missing API keys in .env file")
    print("2. Network connectivity issues")
    print("3. Rate limiting from APIs")
    print("4. Silent failures in the pipeline")

if __name__ == "__main__":
    test_simple_data_fetch()

