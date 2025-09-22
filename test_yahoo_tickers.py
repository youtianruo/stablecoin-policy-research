#!/usr/bin/env python3
"""
Test Yahoo Finance ticker symbols for stablecoins.
"""

import yfinance as yf
import pandas as pd

def test_ticker(ticker_symbol):
    """Test if a ticker symbol works on Yahoo Finance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="5d")
        
        if hist.empty:
            return False, "Empty data"
        
        # Check if we have price data
        if hist['Close'].isna().all():
            return False, "All prices are NaN"
        
        # Check if we have recent data
        recent_price = hist['Close'].dropna().iloc[-1]
        return True, f"Price: ${recent_price:.4f}"
        
    except Exception as e:
        return False, str(e)

def main():
    """Test various ticker symbols for stablecoins."""
    
    print("🔍 Testing Yahoo Finance Ticker Symbols")
    print("=" * 50)
    
    # Test various possible ticker symbols
    test_symbols = [
        "USDT-USD",
        "USDC-USD", 
        "DAI-USD",
        "BTC-USD",
        "BUSD-USD",
        "FRAX-USD",
        "LUSD-USD",
        "TUSD-USD",
        "USDP-USD",
        # Alternative formats
        "USDT",
        "USDC",
        "DAI",
        "BTC",
        # Crypto pairs
        "USDTUSDT",
        "USDCUSDT",
        "DAIUSDT"
    ]
    
    working_symbols = []
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")
        success, message = test_ticker(symbol)
        
        if success:
            print(f"✅ {symbol}: {message}")
            working_symbols.append(symbol)
        else:
            print(f"❌ {symbol}: {message}")
    
    print(f"\n🎯 Working ticker symbols:")
    for symbol in working_symbols:
        print(f"   - {symbol}")
    
    print(f"\n📋 Summary:")
    print(f"   Working: {len(working_symbols)}")
    print(f"   Total tested: {len(test_symbols)}")

if __name__ == "__main__":
    main()
