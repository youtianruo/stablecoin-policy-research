#!/usr/bin/env python3
"""
Simple debug script for ingest pipeline.
"""

import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_ingest():
    """Debug the ingest pipeline step by step."""
    
    print("🔍 Debugging Ingest Pipeline")
    print("=" * 40)
    
    try:
        # Test 1: Import config
        print("\n1️⃣ Testing config loading...")
        from utils.config import load_config
        config = load_config('configs/config.yaml')
        print("✅ Config loaded successfully")
        
        # Test 2: Import pipeline
        print("\n2️⃣ Testing pipeline import...")
        from pipelines.run_ingest import fetch_stablecoin_data
        print("✅ Pipeline imported successfully")
        
        # Test 3: Test data fetching
        print("\n3️⃣ Testing data fetching...")
        start_date = '2024-01-01'
        end_date = '2024-01-07'
        
        print(f"   Date range: {start_date} to {end_date}")
        
        stablecoin_data = fetch_stablecoin_data(config, start_date, end_date)
        
        if stablecoin_data:
            print("✅ Data fetching successful!")
            for key, df in stablecoin_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    print(f"   {key}: {df.shape}")
                    print(f"   Columns: {list(df.columns)}")
                    print(f"   Date range: {df.index.min()} to {df.index.max()}")
                else:
                    print(f"   {key}: Empty or not DataFrame")
        else:
            print("❌ No data returned")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Debug completed!")

if __name__ == "__main__":
    debug_ingest()
