"""
CoinGecko API data fetcher for stablecoin prices and volumes.
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)


class CoinGeckoFetcher:
    """
    Fetches stablecoin data from CoinGecko API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinGecko fetcher.
        
        Args:
            api_key: CoinGecko API key for higher rate limits
        """
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'x-cg-demo-api-key': api_key})
    
    def get_stablecoin_prices(
        self, 
        coin_ids: List[str], 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        vs_currency: str = "usd"
    ) -> pd.DataFrame:
        """
        Fetch historical prices for stablecoins.
        
        Args:
            coin_ids: List of CoinGecko coin IDs
            start_date: Start date for data
            end_date: End date for data
            vs_currency: Currency to get prices in
            
        Returns:
            DataFrame with prices for each stablecoin
        """
        prices_data = {}
        
        for coin_id in coin_ids:
            try:
                logger.info(f"Fetching prices for {coin_id}")
                prices = self._get_coin_prices(coin_id, start_date, end_date, vs_currency)
                prices_data[coin_id] = prices
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching prices for {coin_id}: {e}")
                continue
        
        # Combine into DataFrame
        if prices_data:
            df = pd.DataFrame(prices_data)
            df.index = pd.to_datetime(df.index)
            return df
        else:
            return pd.DataFrame()
    
    def _get_coin_prices(
        self, 
        coin_id: str, 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        vs_currency: str = "usd"
    ) -> pd.Series:
        """
        Get historical prices for a single coin.
        """
        # Convert dates to timestamps
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        url = f"{self.base_url}/coins/{coin_id}/market_chart/range"
        params = {
            'vs_currency': vs_currency,
            'from': start_timestamp,
            'to': end_timestamp
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        prices = data['prices']
        
        # Convert to Series
        timestamps = [pd.to_datetime(price[0], unit='ms') for price in prices]
        values = [price[1] for price in prices]
        
        return pd.Series(values, index=timestamps, name=coin_id)
    
    def get_stablecoin_volumes(
        self, 
        coin_ids: List[str], 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        vs_currency: str = "usd"
    ) -> pd.DataFrame:
        """
        Fetch historical volumes for stablecoins.
        
        Args:
            coin_ids: List of CoinGecko coin IDs
            start_date: Start date for data
            end_date: End date for data
            vs_currency: Currency to get volumes in
            
        Returns:
            DataFrame with volumes for each stablecoin
        """
        volumes_data = {}
        
        for coin_id in coin_ids:
            try:
                logger.info(f"Fetching volumes for {coin_id}")
                volumes = self._get_coin_volumes(coin_id, start_date, end_date, vs_currency)
                volumes_data[coin_id] = volumes
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching volumes for {coin_id}: {e}")
                continue
        
        # Combine into DataFrame
        if volumes_data:
            df = pd.DataFrame(volumes_data)
            df.index = pd.to_datetime(df.index)
            return df
        else:
            return pd.DataFrame()
    
    def _get_coin_volumes(
        self, 
        coin_id: str, 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        vs_currency: str = "usd"
    ) -> pd.Series:
        """
        Get historical volumes for a single coin.
        """
        # Convert dates to timestamps
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        url = f"{self.base_url}/coins/{coin_id}/market_chart/range"
        params = {
            'vs_currency': vs_currency,
            'from': start_timestamp,
            'to': end_timestamp
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        volumes = data['total_volumes']
        
        # Convert to Series
        timestamps = [pd.to_datetime(volume[0], unit='ms') for volume in volumes]
        values = [volume[1] for volume in volumes]
        
        return pd.Series(values, index=timestamps, name=coin_id)
    
    def get_stablecoin_market_caps(
        self, 
        coin_ids: List[str], 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        vs_currency: str = "usd"
    ) -> pd.DataFrame:
        """
        Fetch historical market caps for stablecoins.
        
        Args:
            coin_ids: List of CoinGecko coin IDs
            start_date: Start date for data
            end_date: End date for data
            vs_currency: Currency to get market caps in
            
        Returns:
            DataFrame with market caps for each stablecoin
        """
        market_caps_data = {}
        
        for coin_id in coin_ids:
            try:
                logger.info(f"Fetching market caps for {coin_id}")
                market_caps = self._get_coin_market_caps(coin_id, start_date, end_date, vs_currency)
                market_caps_data[coin_id] = market_caps
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching market caps for {coin_id}: {e}")
                continue
        
        # Combine into DataFrame
        if market_caps_data:
            df = pd.DataFrame(market_caps_data)
            df.index = pd.to_datetime(df.index)
            return df
        else:
            return pd.DataFrame()
    
    def _get_coin_market_caps(
        self, 
        coin_id: str, 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        vs_currency: str = "usd"
    ) -> pd.Series:
        """
        Get historical market caps for a single coin.
        """
        # Convert dates to timestamps
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        url = f"{self.base_url}/coins/{coin_id}/market_chart/range"
        params = {
            'vs_currency': vs_currency,
            'from': start_timestamp,
            'to': end_timestamp
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        market_caps = data['market_caps']
        
        # Convert to Series
        timestamps = [pd.to_datetime(market_cap[0], unit='ms') for market_cap in market_caps]
        values = [market_cap[1] for market_cap in market_caps]
        
        return pd.Series(values, index=timestamps, name=coin_id)
    
    def get_current_stablecoin_data(self, coin_ids: List[str]) -> pd.DataFrame:
        """
        Get current market data for stablecoins.
        
        Args:
            coin_ids: List of CoinGecko coin IDs
            
        Returns:
            DataFrame with current market data
        """
        coin_ids_str = ','.join(coin_ids)
        url = f"{self.base_url}/simple/price"
        params = {
            'ids': coin_ids_str,
            'vs_currencies': 'usd',
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true'
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame
        df_data = []
        for coin_id, coin_data in data.items():
            df_data.append({
                'coin_id': coin_id,
                'price_usd': coin_data['usd'],
                'market_cap_usd': coin_data.get('usd_market_cap'),
                'volume_24h_usd': coin_data.get('usd_24h_vol'),
                'change_24h': coin_data.get('usd_24h_change')
            })
        
        return pd.DataFrame(df_data)
