"""
On-chain data fetcher for blockchain metrics and stablecoin supply data.
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import time
import logging
from web3 import Web3

logger = logging.getLogger(__name__)


class OnChainDataFetcher:
    """
    Fetches on-chain data from Etherscan, Dune Analytics, and other blockchain APIs.
    """
    
    def __init__(self, etherscan_api_key: Optional[str] = None, dune_api_key: Optional[str] = None):
        """
        Initialize on-chain data fetcher.
        
        Args:
            etherscan_api_key: Etherscan API key
            dune_api_key: Dune Analytics API key
        """
        self.etherscan_api_key = etherscan_api_key
        self.dune_api_key = dune_api_key
        
        self.etherscan_base_url = "https://api.etherscan.io/api"
        self.dune_base_url = "https://api.dune.com/api/v1"
        
        self.session = requests.Session()
        
        # Stablecoin contract addresses (Ethereum mainnet)
        self.stablecoin_contracts = {
            'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
            'USDC': '0xA0b86a33E6441b8c4C8C0E4A8c5A8F4A8c5A8F4A',  # Placeholder
            'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
            'BUSD': '0x4Fabb145d64652a948d72533023f6E7A623C5C85',
            'FRAX': '0x853d955aCEf822Db058eb8505911ED77F175b99e',
            'LUSD': '0x5f98805A4E8be255a32880FDeC7F6728C6568bA0'
        }
    
    def get_stablecoin_supply(
        self, 
        stablecoin: str, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime]
    ) -> pd.Series:
        """
        Fetch stablecoin supply data from Etherscan.
        
        Args:
            stablecoin: Stablecoin symbol (e.g., 'USDT', 'USDC')
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Series with stablecoin supply
        """
        if stablecoin not in self.stablecoin_contracts:
            logger.error(f"Unknown stablecoin: {stablecoin}")
            return pd.Series()
        
        contract_address = self.stablecoin_contracts[stablecoin]
        logger.info(f"Fetching supply data for {stablecoin} ({contract_address})")
        
        try:
            # Get current total supply
            current_supply = self._get_token_total_supply(contract_address)
            
            # For historical data, we would need to query at specific block heights
            # This is a simplified version - in practice, you'd need to:
            # 1. Get block numbers for each date
            # 2. Query supply at each block
            # 3. Handle rate limiting
            
            # For now, return current supply as a constant series
            # In practice, implement historical supply tracking
            dates = pd.date_range(start_date, end_date, freq='D')
            supply_series = pd.Series([current_supply] * len(dates), index=dates, name=f'{stablecoin}_supply')
            
            return supply_series
            
        except Exception as e:
            logger.error(f"Error fetching supply data for {stablecoin}: {e}")
            return pd.Series()
    
    def _get_token_total_supply(self, contract_address: str) -> float:
        """
        Get total supply of a token from Etherscan.
        """
        params = {
            'module': 'stats',
            'action': 'tokensupply',
            'contractaddress': contract_address,
            'apikey': self.etherscan_api_key
        }
        
        response = self.session.get(self.etherscan_base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == '1':
            # Convert from wei to token units (assuming 18 decimals)
            supply_wei = int(data['result'])
            supply_tokens = supply_wei / 1e18
            return supply_tokens
        else:
            logger.error(f"Etherscan API error: {data.get('message', 'Unknown error')}")
            return 0.0
    
    def get_stablecoin_transfers(
        self, 
        stablecoin: str, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime],
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Fetch stablecoin transfer data from Etherscan.
        
        Args:
            stablecoin: Stablecoin symbol
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum number of transfers to fetch
            
        Returns:
            DataFrame with transfer data
        """
        if stablecoin not in self.stablecoin_contracts:
            logger.error(f"Unknown stablecoin: {stablecoin}")
            return pd.DataFrame()
        
        contract_address = self.stablecoin_contracts[stablecoin]
        logger.info(f"Fetching transfer data for {stablecoin}")
        
        try:
            params = {
                'module': 'account',
                'action': 'tokentx',
                'contractaddress': contract_address,
                'startblock': 0,
                'endblock': 99999999,
                'sort': 'desc',
                'apikey': self.etherscan_api_key
            }
            
            response = self.session.get(self.etherscan_base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == '1':
                transfers = data['result']
                
                # Convert to DataFrame
                transfer_data = []
                for transfer in transfers[:limit]:  # Limit results
                    transfer_data.append({
                        'hash': transfer['hash'],
                        'from': transfer['from'],
                        'to': transfer['to'],
                        'value': float(transfer['value']) / 1e18,  # Convert from wei
                        'timestamp': pd.to_datetime(int(transfer['timeStamp']), unit='s'),
                        'block_number': int(transfer['blockNumber'])
                    })
                
                df = pd.DataFrame(transfer_data)
                df = df.set_index('timestamp')
                
                # Filter by date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                return df
            else:
                logger.error(f"Etherscan API error: {data.get('message', 'Unknown error')}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching transfer data for {stablecoin}: {e}")
            return pd.DataFrame()
    
    def get_dex_volume(
        self, 
        stablecoin: str, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime]
    ) -> pd.Series:
        """
        Fetch DEX trading volume for stablecoin.
        
        Args:
            stablecoin: Stablecoin symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Series with DEX volume
        """
        logger.info(f"Fetching DEX volume for {stablecoin}")
        
        # This would typically use Dune Analytics API
        # For now, return empty series
        # In practice, you'd query Dune for trading volume data
        
        dates = pd.date_range(start_date, end_date, freq='D')
        volume_series = pd.Series([0.0] * len(dates), index=dates, name=f'{stablecoin}_dex_volume')
        
        return volume_series
    
    def get_stablecoin_holders(
        self, 
        stablecoin: str, 
        top_n: int = 100
    ) -> pd.DataFrame:
        """
        Get top holders of a stablecoin.
        
        Args:
            stablecoin: Stablecoin symbol
            top_n: Number of top holders to return
            
        Returns:
            DataFrame with holder information
        """
        if stablecoin not in self.stablecoin_contracts:
            logger.error(f"Unknown stablecoin: {stablecoin}")
            return pd.DataFrame()
        
        contract_address = self.stablecoin_contracts[stablecoin]
        logger.info(f"Fetching top {top_n} holders for {stablecoin}")
        
        # This would typically use a service like Etherscan or Alchemy
        # For now, return empty DataFrame
        # In practice, you'd query for token holder data
        
        return pd.DataFrame()
    
    def get_peg_stability_metrics(
        self, 
        stablecoin: str, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        Calculate peg stability metrics from on-chain data.
        
        Args:
            stablecoin: Stablecoin symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with peg stability metrics
        """
        logger.info(f"Calculating peg stability metrics for {stablecoin}")
        
        # This would combine multiple on-chain metrics:
        # - Supply changes
        # - Large transfer activity
        # - DEX trading patterns
        # - Arbitrage opportunities
        
        # For now, return empty DataFrame
        # In practice, you'd implement comprehensive peg analysis
        
        dates = pd.date_range(start_date, end_date, freq='D')
        metrics_df = pd.DataFrame(index=dates)
        
        return metrics_df
    
    def get_all_stablecoin_data(
        self, 
        stablecoins: List[str], 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch all on-chain data for multiple stablecoins.
        
        Args:
            stablecoins: List of stablecoin symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary with all on-chain data
        """
        logger.info(f"Fetching all on-chain data for {stablecoins}")
        
        all_data = {}
        
        for stablecoin in stablecoins:
            stablecoin_data = {}
            
            # Fetch different types of data
            stablecoin_data['supply'] = self.get_stablecoin_supply(stablecoin, start_date, end_date)
            stablecoin_data['transfers'] = self.get_stablecoin_transfers(stablecoin, start_date, end_date)
            stablecoin_data['dex_volume'] = self.get_dex_volume(stablecoin, start_date, end_date)
            stablecoin_data['peg_metrics'] = self.get_peg_stability_metrics(stablecoin, start_date, end_date)
            
            all_data[stablecoin] = stablecoin_data
        
        return all_data
