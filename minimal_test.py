#!/usr/bin/env python3
"""
Minimal test to fetch a small amount of data and verify the pipeline works.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def minimal_data_test():
    """Fetch minimal data to test the pipeline."""
    
    print("🧪 Minimal Data Test")
    print("=" * 30)
    
    try:
        # Load config
        from utils.config import load_config, ensure_directory
        from utils.io import save_data
        
        config = load_config('configs/config.yaml')
        
        # Create directories
        ensure_directory(config['data']['raw_dir'])
        ensure_directory(config['data']['processed_dir'])
        
        print(f"✅ Config loaded and directories created")
        
        # Test CoinGecko with minimal data
        from data.fetch_coingecko import CoinGeckoFetcher
        
        fetcher = CoinGeckoFetcher()
        
        # Test with just USDT for 2 days
        start_date = '2024-01-01'
        end_date = '2024-01-02'
        
        print(f"📊 Fetching USDT prices for {start_date} to {end_date}")
        
        prices = fetcher.get_stablecoin_prices(['tether'], start_date, end_date)
        
        if not prices.empty:
            print(f"✅ Successfully fetched {len(prices)} price points")
            print(f"   Shape: {prices.shape}")
            print(f"   Columns: {list(prices.columns)}")
            print(f"   Date range: {prices.index.min()} to {prices.index.max()}")
            print(f"   Sample prices: {prices.head()}")
            
            # Save the data
            save_data(prices, 'test_stablecoin_prices', config['data']['processed_dir'])
            print(f"✅ Data saved to processed directory")
            
            # Verify file exists
            test_file = os.path.join(config['data']['processed_dir'], 'test_stablecoin_prices.parquet')
            if os.path.exists(test_file):
                file_size = os.path.getsize(test_file)
                print(f"✅ File verified: {test_file} ({file_size} bytes)")
            else:
                print(f"❌ File not found: {test_file}")
                
        else:
            print(f"❌ No price data returned")
            
            # Try with a different approach
            print("🔄 Trying alternative approach...")
            
            import requests
            
            # Direct API call to test
            url = "https://api.coingecko.com/api/v3/coins/tether/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': int(pd.Timestamp(start_date).timestamp()),
                'to': int(pd.Timestamp(end_date).timestamp())
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'prices' in data and data['prices']:
                    print(f"✅ Direct API call successful")
                    print(f"   Got {len(data['prices'])} price points")
                    
                    # Convert to DataFrame
                    prices_data = []
                    for price_point in data['prices']:
                        prices_data.append({
                            'timestamp': pd.to_datetime(price_point[0], unit='ms'),
                            'price': price_point[1]
                        })
                    
                    df = pd.DataFrame(prices_data)
                    df = df.set_index('timestamp')
                    df.columns = ['tether']
                    
                    print(f"✅ Created DataFrame: {df.shape}")
                    print(f"   Sample data: {df.head()}")
                    
                    # Save this data
                    save_data(df, 'test_stablecoin_prices_direct', config['data']['processed_dir'])
                    print(f"✅ Direct API data saved")
                    
                else:
                    print(f"❌ No price data in API response")
            else:
                print(f"❌ Direct API call failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Minimal test completed!")
    print("\nIf this test passes, the issue might be:")
    print("1. Date range too large causing timeouts")
    print("2. Too many stablecoins being fetched at once")
    print("3. Rate limiting from CoinGecko")
    print("4. Silent failures in the full pipeline")

if __name__ == "__main__":
    minimal_data_test()

