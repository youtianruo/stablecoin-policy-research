#!/usr/bin/env python3
"""
Test the fixed pipeline to verify API fetching works.
"""

import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_fixed_pipeline():
    """Test the fixed pipeline."""
    
    print("🧪 Testing Fixed Pipeline")
    print("=" * 40)
    
    try:
        # Test the CoinGecko fetcher directly
        print("\n1️⃣ Testing CoinGecko fetcher directly...")
        from data.fetch_coingecko import fetch_series
        
        # Test with USDT for 30 days
        df = fetch_series('USDT', vs_currency='usd', days=30)
        
        if not df.empty:
            print(f"✅ Direct fetcher works!")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"   Sample data:")
            print(df.head().to_string(index=False))
        else:
            print(f"❌ Direct fetcher returned empty data")
            return
        
    except Exception as e:
        print(f"❌ Direct fetcher test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        # Test the pipeline function
        print("\n2️⃣ Testing pipeline function...")
        from pipelines.run_ingest import fetch_stablecoin_data
        from utils.config import load_config
        
        config = load_config('configs/config.yaml')
        
        # Test with small date range
        start_date = '2024-01-01'
        end_date = '2024-01-31'
        
        stablecoin_data = fetch_stablecoin_data(config, start_date, end_date)
        
        if stablecoin_data:
            print(f"✅ Pipeline function works!")
            for key, df in stablecoin_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    print(f"   {key}: {df.shape}")
                    print(f"   Columns: {list(df.columns)}")
                    print(f"   Date range: {df.index.min()} to {df.index.max()}")
                else:
                    print(f"   {key}: Empty or not DataFrame")
        else:
            print(f"❌ Pipeline function returned no data")
            
    except Exception as e:
        print(f"❌ Pipeline function test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Fixed pipeline test completed!")
    print("\nIf both tests pass, the API fetching should now work!")
    print("Try running: run_all_windows.bat")

if __name__ == "__main__":
    test_fixed_pipeline()

