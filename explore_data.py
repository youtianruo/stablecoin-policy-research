#!/usr/bin/env python3
"""
Data exploration tool for stablecoin policy research.
This script helps you understand what data is available and how to interpret it.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def explore_data_structure():
    """Explore the data directory structure and file contents."""
    
    print("🔍 Stablecoin Policy Research - Data Explorer")
    print("=" * 60)
    
    # Check data directories
    data_dirs = ['data/raw', 'data/interim', 'data/processed']
    
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"\n📁 {dir_path}/ ({len(files)} files)")
            print("-" * 40)
            
            for file in sorted(files):
                file_path = os.path.join(dir_path, file)
                if file.endswith('.parquet'):
                    try:
                        df = pd.read_parquet(file_path)
                        print(f"📊 {file}")
                        print(f"   Shape: {df.shape}")
                        print(f"   Columns: {list(df.columns)}")
                        if 'timestamp' in df.columns or 'date' in df.columns:
                            date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
                            print(f"   Date range: {df[date_col].min()} to {df[date_col].max()}")
                        print()
                    except Exception as e:
                        print(f"❌ {file}: Error reading - {e}")
                elif file.endswith('.json'):
                    print(f"📄 {file} (JSON data)")
                else:
                    print(f"📄 {file}")

def analyze_stablecoin_prices():
    """Analyze the main stablecoin price data."""
    
    print("\n💰 Stablecoin Price Analysis")
    print("=" * 40)
    
    try:
        df = pd.read_parquet('data/processed/stablecoin_prices.parquet')
        
        print(f"📊 Dataset Shape: {df.shape}")
        print(f"📅 Date Range: {df.index.min()} to {df.index.max()}")
        print(f"🪙 Stablecoins: {list(df.columns)}")
        
        print(f"\n📈 Current Prices:")
        latest_prices = df.iloc[-1]
        for coin, price in latest_prices.items():
            print(f"   {coin}: ${price:.6f}")
        
        print(f"\n📊 Price Statistics:")
        stats = df.describe()
        print(stats.round(6))
        
        print(f"\n🔍 Peg Deviation Analysis:")
        for coin in df.columns:
            peg_dev = abs(df[coin] - 1.0)
            max_dev = peg_dev.max()
            avg_dev = peg_dev.mean()
            print(f"   {coin}: Max deviation {max_dev:.6f}, Avg deviation {avg_dev:.6f}")
            
    except Exception as e:
        print(f"❌ Error analyzing prices: {e}")

def analyze_volatility():
    """Analyze volatility patterns."""
    
    print("\n📈 Volatility Analysis")
    print("=" * 30)
    
    try:
        df = pd.read_parquet('data/processed/volatility.parquet')
        
        print(f"📊 Volatility Dataset Shape: {df.shape}")
        print(f"📅 Date Range: {df.index.min()} to {df.index.max()}")
        
        print(f"\n📊 Volatility by Stablecoin:")
        for col in df.columns:
            if col != 'timestamp':
                avg_vol = df[col].mean()
                max_vol = df[col].max()
                print(f"   {col}: Avg {avg_vol:.6f}, Max {max_vol:.6f}")
                
    except Exception as e:
        print(f"❌ Error analyzing volatility: {e}")

def analyze_peg_deviations():
    """Analyze peg deviation patterns."""
    
    print("\n🎯 Peg Deviation Analysis")
    print("=" * 35)
    
    try:
        df = pd.read_parquet('data/processed/peg_deviations.parquet')
        
        print(f"📊 Peg Deviation Dataset Shape: {df.shape}")
        print(f"📅 Date Range: {df.index.min()} to {df.index.max()}")
        
        print(f"\n🎯 Deviation Statistics:")
        for col in df.columns:
            if col != 'timestamp':
                deviations = df[col]
                positive_dev = (deviations > 0).sum()
                negative_dev = (deviations < 0).sum()
                max_dev = deviations.max()
                min_dev = deviations.min()
                print(f"   {col}:")
                print(f"     Positive deviations: {positive_dev} ({positive_dev/len(deviations)*100:.1f}%)")
                print(f"     Negative deviations: {negative_dev} ({negative_dev/len(deviations)*100:.1f}%)")
                print(f"     Max deviation: {max_dev:.6f}")
                print(f"     Min deviation: {min_dev:.6f}")
                
    except Exception as e:
        print(f"❌ Error analyzing peg deviations: {e}")

def show_data_interpretation_guide():
    """Show guide for interpreting the data."""
    
    print("\n📚 Data Interpretation Guide")
    print("=" * 40)
    
    print("""
🔍 Key Data Files Explained:

📊 stablecoin_prices.parquet
   - Daily closing prices for all stablecoins
   - Values close to 1.0 indicate good peg maintenance
   - Deviations from 1.0 show peg stress

📈 volatility.parquet  
   - Rolling volatility measures for each stablecoin
   - Higher values = more price instability
   - Important for risk assessment

🎯 peg_deviations.parquet
   - Direct measure of how far prices deviate from $1.00
   - Positive = trading above peg (premium)
   - Negative = trading below peg (discount)

📊 returns.parquet
   - Daily returns (price changes) for each stablecoin
   - Should be close to 0 for well-pegged stablecoins
   - Large values indicate peg breaks

📈 market_depth.parquet
   - Measures of market liquidity and depth
   - Higher values = more liquid markets
   - Important for stability assessment

🔍 rolling_correlation.parquet
   - How stablecoins move together over time
   - High correlation = systemic risk
   - Low correlation = independent behavior

📊 volatility_regime.parquet
   - Identifies periods of high/low volatility
   - Helps identify stress periods
   - Useful for event study analysis

🎯 Key Metrics to Watch:
   - Price stability (close to $1.00)
   - Volatility levels (lower is better)
   - Correlation patterns (systemic vs independent)
   - Peg deviation frequency and magnitude
   - Market depth and liquidity measures
""")

def main():
    """Main exploration function."""
    
    # Check if data exists
    if not os.path.exists('data/processed'):
        print("❌ No processed data found. Run the pipeline first:")
        print("   run_all_windows.bat")
        return
    
    # Run all analysis functions
    explore_data_structure()
    analyze_stablecoin_prices()
    analyze_volatility()
    analyze_peg_deviations()
    show_data_interpretation_guide()
    
    print("\n🎉 Data exploration complete!")
    print("\n💡 Next steps:")
    print("   1. Run Jupyter notebooks for interactive analysis")
    print("   2. Check notebooks/00_exploration.ipynb for visualizations")
    print("   3. Use notebooks/10_figures.ipynb for publication-ready plots")

if __name__ == "__main__":
    main()
