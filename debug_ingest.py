#!/usr/bin/env python3
"""
Debug script for data ingestion issues.
"""

import sys
import os
import traceback
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_data_ingestion():
    """Debug data ingestion step by step."""
    
    print("🔍 Debugging Data Ingestion Pipeline")
    print("=" * 50)
    
    try:
        # 1. Test configuration loading
        print("\n1️⃣ Testing configuration loading...")
        from utils.config import load_config
        config = load_config('configs/config.yaml')
        print(f"✅ Config loaded successfully")
        print(f"   - Raw dir: {config['data']['raw_dir']}")
        print(f"   - Processed dir: {config['data']['processed_dir']}")
        print(f"   - Start date: {config.get('analysis', {}).get('start_date', 'Not set')}")
        print(f"   - End date: {config.get('analysis', {}).get('end_date', 'Not set')}")
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        traceback.print_exc()
        return
    
    try:
        # 2. Test directory creation
        print("\n2️⃣ Testing directory creation...")
        from utils.config import ensure_directory
        ensure_directory(config['data']['raw_dir'])
        ensure_directory(config['data']['interim_dir'])
        ensure_directory(config['data']['processed_dir'])
        print(f"✅ Directories created successfully")
        
    except Exception as e:
        print(f"❌ Directory creation failed: {e}")
        traceback.print_exc()
        return
    
    try:
        # 3. Test CoinGecko fetcher
        print("\n3️⃣ Testing CoinGecko fetcher...")
        from data.fetch_coingecko import CoinGeckoFetcher
        
        api_keys = config.get('api_keys', {})
        coingecko_key = api_keys.get('coingecko')
        
        fetcher = CoinGeckoFetcher(coingecko_key)
        print(f"✅ CoinGecko fetcher initialized")
        print(f"   - API key present: {'Yes' if coingecko_key else 'No'}")
        
        # Test with a small date range
        start_date = '2024-01-01'
        end_date = '2024-01-02'
        
        stablecoins = config.get('stablecoins', {})
        tickers = stablecoins.get('tickers', ['USDT', 'USDC'])
        coingecko_ids = stablecoins.get('coingecko_ids', {})
        
        print(f"   - Tickers: {tickers}")
        print(f"   - CoinGecko IDs: {coingecko_ids}")
        
        # Convert tickers to CoinGecko IDs
        coin_ids = []
        for ticker in tickers[:2]:  # Test with first 2
            if ticker in coingecko_ids:
                coin_ids.append(coingecko_ids[ticker])
            else:
                print(f"   ⚠️ No CoinGecko ID for {ticker}")
        
        if coin_ids:
            print(f"   - Testing with IDs: {coin_ids}")
            prices = fetcher.get_stablecoin_prices(coin_ids, start_date, end_date)
            print(f"✅ CoinGecko data fetched: {prices.shape if not prices.empty else 'Empty'}")
            
            if not prices.empty:
                print(f"   - Date range: {prices.index.min()} to {prices.index.max()}")
                print(f"   - Columns: {list(prices.columns)}")
            else:
                print("   ⚠️ No price data returned")
        else:
            print("   ⚠️ No valid CoinGecko IDs found")
        
    except Exception as e:
        print(f"❌ CoinGecko fetcher failed: {e}")
        traceback.print_exc()
    
    try:
        # 4. Test Fed fetcher
        print("\n4️⃣ Testing Fed data fetcher...")
        from data.fetch_fed import FedDataFetcher
        
        fed_fetcher = FedDataFetcher()
        print(f"✅ Fed fetcher initialized")
        
        # Test with small date range
        start_date = '2024-01-01'
        end_date = '2024-01-31'
        
        policy_events = fed_fetcher.get_policy_events(start_date, end_date)
        print(f"✅ Fed data fetched: {policy_events.shape if not policy_events.empty else 'Empty'}")
        
        if not policy_events.empty:
            print(f"   - Event types: {policy_events['event_type'].unique()}")
            print(f"   - Date range: {policy_events['date'].min()} to {policy_events['date'].max()}")
        else:
            print("   ⚠️ No policy events returned")
        
    except Exception as e:
        print(f"❌ Fed fetcher failed: {e}")
        traceback.print_exc()
    
    try:
        # 5. Test Macro fetcher
        print("\n5️⃣ Testing Macro data fetcher...")
        from data.fetch_macro import MacroDataFetcher
        
        fred_key = api_keys.get('fred')
        if fred_key:
            macro_fetcher = MacroDataFetcher(fred_key)
            print(f"✅ Macro fetcher initialized with FRED key")
            
            # Test with small date range
            start_date = '2024-01-01'
            end_date = '2024-01-31'
            
            fed_funds = macro_fetcher.get_fed_funds_rate(start_date, end_date)
            print(f"✅ Fed funds rate fetched: {fed_funds.shape if not fed_funds.empty else 'Empty'}")
            
            if not fed_funds.empty:
                print(f"   - Date range: {fed_funds.index.min()} to {fed_funds.index.max()}")
                print(f"   - Sample values: {fed_funds.head().values}")
        else:
            print("⚠️ No FRED API key found - skipping macro data test")
        
    except Exception as e:
        print(f"❌ Macro fetcher failed: {e}")
        traceback.print_exc()
    
    try:
        # 6. Test data saving
        print("\n6️⃣ Testing data saving...")
        from utils.io import save_data
        
        # Create test data
        import pandas as pd
        import numpy as np
        
        test_data = pd.DataFrame({
            'test_col': [1, 2, 3, 4, 5]
        }, index=pd.date_range('2024-01-01', periods=5))
        
        save_data(test_data, 'test_data', config['data']['processed_dir'], file_format='parquet')
        print(f"✅ Test data saved successfully")
        
        # Check if file exists
        test_file = os.path.join(config['data']['processed_dir'], 'test_data.parquet')
        if os.path.exists(test_file):
            print(f"✅ Test file exists: {test_file}")
            file_size = os.path.getsize(test_file)
            print(f"   - File size: {file_size} bytes")
        else:
            print(f"❌ Test file not found: {test_file}")
        
    except Exception as e:
        print(f"❌ Data saving failed: {e}")
        traceback.print_exc()
    
    print("\n🏁 Debug completed!")
    print("\nNext steps:")
    print("1. Check the output above for any ❌ errors")
    print("2. If API keys are missing, add them to .env file")
    print("3. If network issues, check your internet connection")
    print("4. If directory issues, check file permissions")

if __name__ == "__main__":
    debug_data_ingestion()

