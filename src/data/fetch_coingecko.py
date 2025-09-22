import os
import time
import math
import json
import logging
from typing import Dict, Any, Optional
import requests
import pandas as pd
import yfinance as yf

BASE_COINGECKO = "https://api.coingecko.com/api/v3"
BASE_COINCAP = "https://api.coincap.io/v2"

# Symbol mappings for different providers
COINGECKO_MAP = {
    "USDT": "tether", 
    "USDC": "usd-coin", 
    "DAI": "dai", 
    "BTC": "bitcoin",
    "BUSD": "binance-usd",
    "FRAX": "frax",
    "LUSD": "liquity-usd",
    "TUSD": "true-usd",
    "USDP": "paxos-standard"
}

YAHOO_MAP = {
    "USDT": "USDT-USD",
    "USDC": "USDC-USD", 
    "DAI": "DAI-USD",
    "BTC": "BTC-USD",
    "BUSD": "BUSD-USD",
    "FRAX": "FRAX-USD",
    "LUSD": "LUSD-USD",
    "TUSD": "TUSD-USD",
    "USDP": "USDP-USD"
}

COINCAP_MAP = {
    "USDT": "tether",
    "USDC": "usd-coin",
    "DAI": "dai",
    "BTC": "bitcoin",
    "BUSD": "binance-usd",
    "FRAX": "frax",
    "LUSD": "liquity-usd",
    "TUSD": "true-usd",
    "USDP": "paxos-standard"
}

HEADERS = {
    "User-Agent": "stablecoin-policy-research/0.1 (+https://example.org)",
    "Accept": "application/json",
}

TIMEOUT = 30  # seconds
MAX_RETRIES = 3
BACKOFF_BASE = 1.8  # exponential backoff

class FetchError(RuntimeError):
    pass

def _request_coingecko(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Make request to CoinGecko API with retry logic."""
    last_err = None
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = BACKOFF_BASE ** i + (0.2 * i)
                logging.warning("CoinGecko HTTP %s. Backing off %.1fs", r.status_code, wait)
                time.sleep(wait)
                continue
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            raise FetchError(f"CoinGecko HTTP {r.status_code} :: {detail}")
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            wait = BACKOFF_BASE ** i + (0.2 * i)
            logging.warning("CoinGecko network error: %s. Backing off %.1fs", e, wait)
            time.sleep(wait)
    raise FetchError(f"CoinGecko failed after {MAX_RETRIES} retries: {last_err}")

def _request_coincap(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Make request to CoinCap API with retry logic."""
    last_err = None
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = BACKOFF_BASE ** i + (0.2 * i)
                logging.warning("CoinCap HTTP %s. Backing off %.1fs", r.status_code, wait)
                time.sleep(wait)
                continue
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            raise FetchError(f"CoinCap HTTP {r.status_code} :: {detail}")
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            wait = BACKOFF_BASE ** i + (0.2 * i)
            logging.warning("CoinCap network error: %s. Backing off %.1fs", e, wait)
            time.sleep(wait)
    raise FetchError(f"CoinCap failed after {MAX_RETRIES} retries: {last_err}")

def fetch_coingecko(symbol: str, vs_currency: str = "usd", days: int = 3650) -> pd.DataFrame:
    """Fetch data from CoinGecko."""
    if symbol not in COINGECKO_MAP:
        raise ValueError(f"Unknown symbol {symbol} for CoinGecko. Known: {list(COINGECKO_MAP)}")
    
    if days > 3650:
        logging.warning(f"Days parameter {days} exceeds CoinGecko limit (3650). Using 3650.")
        days = 3650
    
    coin_id = COINGECKO_MAP[symbol]
    url = f"{BASE_COINGECKO}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "precision": "full"}
    
    data = _request_coingecko(url, params)
    
    if "prices" not in data or "total_volumes" not in data:
        raise FetchError(f"CoinGecko unexpected response for {symbol}: {json.dumps(data)[:300]}")
    
    prices = pd.DataFrame(data["prices"], columns=["ts", "price"])
    vols = pd.DataFrame(data["total_volumes"], columns=["ts", "volume"])
    out = prices.merge(vols, on="ts", how="left")
    out["timestamp"] = pd.to_datetime(out["ts"], unit="ms", utc=True)
    out["symbol"] = symbol
    out["provider"] = "coingecko"
    return out[["timestamp", "symbol", "price", "volume", "provider"]]

def fetch_yahoo(symbol: str, vs_currency: str = "usd", days: int = 3650) -> pd.DataFrame:
    """Fetch data from Yahoo Finance."""
    if symbol not in YAHOO_MAP:
        raise ValueError(f"Unknown symbol {symbol} for Yahoo Finance. Known: {list(YAHOO_MAP)}")
    
    ticker_symbol = YAHOO_MAP[symbol]
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Use period instead of date range for better reliability
        if days <= 7:
            period = "7d"
        elif days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        elif days <= 365:
            period = "1y"
        elif days <= 730:
            period = "2y"
        else:
            period = "5y"
        
        # Fetch data
        hist = ticker.history(period=period, interval="1d")
        
        if hist.empty:
            raise FetchError(f"Yahoo Finance returned empty data for {symbol}")
        
        # Convert to our format - use reset_index to preserve the data
        df = hist.reset_index()
        df = df.rename(columns={'Date': 'timestamp', 'Close': 'price', 'Volume': 'volume'})
        df["symbol"] = symbol
        df["provider"] = "yahoo"
        
        # Remove rows with NaN prices
        df = df.dropna(subset=['price'])
        
        if df.empty:
            raise FetchError(f"Yahoo Finance returned no valid price data for {symbol}")
        
        return df[["timestamp", "symbol", "price", "volume", "provider"]]
        
    except Exception as e:
        raise FetchError(f"Yahoo Finance error for {symbol}: {e}")

def fetch_coincap(symbol: str, vs_currency: str = "usd", days: int = 3650) -> pd.DataFrame:
    """Fetch data from CoinCap."""
    if symbol not in COINCAP_MAP:
        raise ValueError(f"Unknown symbol {symbol} for CoinCap. Known: {list(COINCAP_MAP)}")
    
    coin_id = COINCAP_MAP[symbol]
    
    # Calculate timestamps
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    url = f"{BASE_COINCAP}/assets/{coin_id}/history"
    params = {
        "interval": "d1",
        "start": start_time,
        "end": end_time
    }
    
    data = _request_coincap(url, params)
    
    if "data" not in data:
        raise FetchError(f"CoinCap unexpected response for {symbol}: {json.dumps(data)[:300]}")
    
    if not data["data"]:
        raise FetchError(f"CoinCap returned empty data for {symbol}")
    
    # Convert to DataFrame
    df = pd.DataFrame(data["data"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["symbol"] = symbol
    df["price"] = pd.to_numeric(df["priceUsd"])
    df["volume"] = pd.to_numeric(df["volumeUsd"])
    df["provider"] = "coincap"
    
    return df[["timestamp", "symbol", "price", "volume", "provider"]]

def fetch_series(symbol: str, vs_currency: str = "usd", days: int = 3650) -> pd.DataFrame:
    """
    Fetch data from multiple providers with fallback.
    Tries: Yahoo Finance → CoinCap (CoinGecko skipped due to rate limits)
    """
    providers = [
        ("Yahoo Finance", fetch_yahoo),
        ("CoinCap", fetch_coincap)
    ]
    
    last_error = None
    
    for provider_name, fetch_func in providers:
        try:
            logging.info(f"Trying {provider_name} for {symbol}...")
            df = fetch_func(symbol, vs_currency, days)
            
            if not df.empty:
                logging.info(f"✅ Successfully fetched {len(df)} data points for {symbol} from {provider_name}")
                return df
            else:
                logging.warning(f"⚠️ {provider_name} returned empty data for {symbol}")
                
        except Exception as e:
            last_error = e
            logging.warning(f"❌ {provider_name} failed for {symbol}: {e}")
            continue
    
    # If all providers failed
    raise FetchError(f"All providers failed for {symbol}. Last error: {last_error}")

def get_market_chart(coin_id: str, vs_currency: str, days: int) -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    url = f"{BASE_COINGECKO}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "precision": "full"}
    return _request_coingecko(url, params)

def main():
    """Allow quick manual run: python -m src.data.fetch_coingecko"""
    symbols = os.environ.get("SYMS", "USDT,USDC,DAI,BTC").split(",")
    days = int(os.environ.get("DAYS", "3650"))
    vs = os.environ.get("VS", "usd")

    frames = []
    for s in symbols:
        s = s.strip().upper()
        logging.info("Fetching %s...", s)
        df = fetch_series(s, vs_currency=vs, days=days)
        frames.append(df)
        time.sleep(1.2)  # be polite

    df_all = pd.concat(frames, ignore_index=True)
    # Print a tiny preview so our .bat capture shows something
    print(df_all.head().to_string(index=False))
    
    # Show provider summary
    if "provider" in df_all.columns:
        provider_counts = df_all["provider"].value_counts()
        print(f"\nProvider usage: {dict(provider_counts)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()