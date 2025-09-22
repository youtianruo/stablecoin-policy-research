"""
Data ingestion pipeline for stablecoin policy research.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, ensure_directory
from utils.io import save_data, load_data
from data.fetch_coingecko import fetch_series
from data.fetch_fed import FedDataFetcher
from data.fetch_macro import MacroDataFetcher
from data.fetch_onchain import OnChainDataFetcher

logger = logging.getLogger(__name__)


def run_data_ingestion(config_path: str = "configs/config.yaml") -> Dict[str, pd.DataFrame]:
    """
    Run complete data ingestion pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with ingested data
    """
    logger.info("Starting data ingestion pipeline")
    
    # Load configuration
    config = load_config(config_path)
    
    # Ensure data directories exist
    ensure_directory(config['data']['raw_dir'])
    ensure_directory(config['data']['interim_dir'])
    ensure_directory(config['data']['processed_dir'])
    
    # Get date range
    # Use a date range that works with CoinGecko free tier (365 days max)
    start_date = config.get('analysis', {}).get('start_date', '2024-01-01')
    end_date = config.get('analysis', {}).get('end_date', datetime.now().strftime('%Y-%m-%d'))
    
    logger.info(f"Data ingestion period: {start_date} to {end_date}")
    
    # Initialize data fetchers
    api_keys = config.get('api_keys', {})
    
    # Note: Hardened CoinGecko fetcher doesn't need API key for basic usage
    # Initialize fetchers (some may fail if API keys are missing)
    fed_fetcher = None
    macro_fetcher = None
    onchain_fetcher = None
    
    try:
        fed_fetcher = FedDataFetcher()
    except Exception as e:
        logger.warning(f"FedDataFetcher initialization failed: {e}")
    
    try:
        macro_fetcher = MacroDataFetcher(api_keys.get('fred'))
    except Exception as e:
        logger.warning(f"MacroDataFetcher initialization failed: {e}")
    
    try:
        onchain_fetcher = OnChainDataFetcher(
            api_keys.get('etherscan'),
            api_keys.get('dune')
        )
    except Exception as e:
        logger.warning(f"OnChainDataFetcher initialization failed: {e}")
    
    # Ingest data
    ingested_data = {}
    
    try:
        # 1. Stablecoin data
        logger.info("Fetching stablecoin data")
        stablecoin_data = fetch_stablecoin_data(
            config, start_date, end_date
        )
        ingested_data.update(stablecoin_data)
        
        # 2. Policy events (optional - requires FRED API key)
        if fed_fetcher is not None:
            try:
                logger.info("Fetching policy events")
                policy_events = fetch_policy_events(
                    fed_fetcher, start_date, end_date
                )
                ingested_data['policy_events'] = policy_events
            except Exception as e:
                logger.warning(f"Skipping policy events: {e}")
                ingested_data['policy_events'] = pd.DataFrame()
        else:
            logger.info("Skipping policy events (FedDataFetcher not available)")
            ingested_data['policy_events'] = pd.DataFrame()
        
        # 3. Macroeconomic data (optional - requires FRED API key)
        if macro_fetcher is not None:
            try:
                logger.info("Fetching macroeconomic data")
                macro_data = fetch_macro_data(
                    macro_fetcher, start_date, end_date
                )
                ingested_data.update(macro_data)
            except Exception as e:
                logger.warning(f"Skipping macro data: {e}")
                ingested_data['macro_data'] = pd.DataFrame()
        else:
            logger.info("Skipping macro data (MacroDataFetcher not available)")
            ingested_data['macro_data'] = pd.DataFrame()
        
        # 4. On-chain data (optional)
        if onchain_fetcher is not None:
            try:
                logger.info("Fetching on-chain data")
                onchain_data = fetch_onchain_data(
                    onchain_fetcher, config, start_date, end_date
                )
                ingested_data.update(onchain_data)
            except Exception as e:
                logger.warning(f"Skipping on-chain data: {e}")
                ingested_data['onchain_data'] = pd.DataFrame()
        else:
            logger.info("Skipping on-chain data (OnChainDataFetcher not available)")
            ingested_data['onchain_data'] = pd.DataFrame()
        
        # Save all data
        logger.info("Saving ingested data")
        save_ingested_data(ingested_data, config)
        
        logger.info("Data ingestion pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data ingestion pipeline: {e}")
        raise
    
    return ingested_data


def fetch_stablecoin_data(
    config: Dict,
    start_date: str,
    end_date: str
) -> Dict[str, pd.DataFrame]:
    """
    Fetch stablecoin market data.
    
    Args:
        config: Configuration dictionary
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Dictionary with stablecoin data
    """
    stablecoin_data = {}
    
    # Get stablecoin configurations
    stablecoins = config.get('stablecoins', {})
    tickers = stablecoins.get('tickers', ['USDT', 'USDC', 'DAI'])
    coingecko_ids = stablecoins.get('coingecko_ids', {})
    
    # Convert tickers to CoinGecko IDs
    coin_ids = []
    for ticker in tickers:
        if ticker in coingecko_ids:
            coin_ids.append(coingecko_ids[ticker])
        else:
            logger.warning(f"No CoinGecko ID found for {ticker}")
    
    if not coin_ids:
        logger.warning("No valid CoinGecko IDs found")
        return stablecoin_data
    
    try:
        # Fetch data using the new hardened fetcher
        logger.info("Fetching stablecoin data using hardened fetcher")
        
        # Calculate days from date range
        from datetime import datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end_dt - start_dt).days
        
        # Fetch data for each ticker
        all_data = []
        for ticker in tickers:
            if ticker in coingecko_ids:
                logger.info(f"Fetching data for {ticker}")
                try:
                    df = fetch_series(ticker, vs_currency="usd", days=days)
                    all_data.append(df)
                    logger.info(f"Successfully fetched {len(df)} data points for {ticker}")
                except Exception as e:
                    logger.error(f"Error fetching {ticker}: {e}")
                    continue
        
        if all_data:
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Separate into prices and volumes
            try:
                prices_df = combined_df.pivot(index='timestamp', columns='symbol', values='price')
                volumes_df = combined_df.pivot(index='timestamp', columns='symbol', values='volume')
                
                # Handle potential NaN values
                prices_df = prices_df.dropna(how='all')  # Remove rows with all NaN
                volumes_df = volumes_df.dropna(how='all')
                
                stablecoin_data['stablecoin_prices'] = prices_df
                stablecoin_data['stablecoin_volumes'] = volumes_df
                
                logger.info(f"Successfully fetched data for {len(prices_df.columns)} stablecoins")
                logger.info(f"Date range: {prices_df.index.min()} to {prices_df.index.max()}")
                logger.info(f"Price data shape: {prices_df.shape}")
                logger.info(f"Volume data shape: {volumes_df.shape}")
                
            except Exception as e:
                logger.error(f"Error processing fetched data: {e}")
                # Fallback: save raw combined data
                stablecoin_data['stablecoin_raw_data'] = combined_df
                logger.info(f"Saved raw data with shape: {combined_df.shape}")
        
    except Exception as e:
        logger.error(f"Error fetching stablecoin data: {e}")
        import traceback
        traceback.print_exc()
    
    return stablecoin_data


def fetch_policy_events(
    fetcher: FedDataFetcher,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetch Federal Reserve policy events.
    
    Args:
        fetcher: Fed data fetcher instance
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with policy events
    """
    try:
        # Fetch all policy events
        policy_events = fetcher.get_policy_events(start_date, end_date)
        
        if not policy_events.empty:
            logger.info(f"Fetched {len(policy_events)} policy events")
        else:
            logger.warning("No policy events fetched")
        
        return policy_events
        
    except Exception as e:
        logger.error(f"Error fetching policy events: {e}")
        return pd.DataFrame()


def fetch_macro_data(
    fetcher: MacroDataFetcher,
    start_date: str,
    end_date: str
) -> Dict[str, pd.DataFrame]:
    """
    Fetch macroeconomic data.
    
    Args:
        fetcher: Macro data fetcher instance
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Dictionary with macro data
    """
    macro_data = {}
    
    try:
        # Fetch all macro data
        all_macro_data = fetcher.get_all_macro_data(start_date, end_date)
        
        # Organize by category
        macro_data['fed_funds_rate'] = all_macro_data.get('fed_funds', pd.Series())
        macro_data['treasury_rates'] = all_macro_data.get('treasury_rates', pd.DataFrame())
        macro_data['inflation_data'] = all_macro_data.get('inflation', pd.DataFrame())
        macro_data['employment_data'] = all_macro_data.get('employment', pd.DataFrame())
        macro_data['gdp_data'] = all_macro_data.get('gdp', pd.DataFrame())
        macro_data['financial_indicators'] = all_macro_data.get('financial', pd.DataFrame())
        
        logger.info("Fetched macroeconomic data")
        
    except Exception as e:
        logger.error(f"Error fetching macro data: {e}")
    
    return macro_data


def fetch_onchain_data(
    fetcher: OnChainDataFetcher,
    config: Dict,
    start_date: str,
    end_date: str
) -> Dict[str, pd.DataFrame]:
    """
    Fetch on-chain data.
    
    Args:
        fetcher: On-chain data fetcher instance
        config: Configuration dictionary
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Dictionary with on-chain data
    """
    onchain_data = {}
    
    try:
        # Get stablecoin tickers
        stablecoins = config.get('stablecoins', {}).get('tickers', ['USDT', 'USDC', 'DAI'])
        
        # Fetch on-chain data for each stablecoin
        all_onchain_data = fetcher.get_all_stablecoin_data(stablecoins, start_date, end_date)
        
        # Organize by type
        for stablecoin, data in all_onchain_data.items():
            if 'supply' in data:
                onchain_data[f'{stablecoin}_supply'] = data['supply']
            if 'transfers' in data:
                onchain_data[f'{stablecoin}_transfers'] = data['transfers']
            if 'dex_volume' in data:
                onchain_data[f'{stablecoin}_dex_volume'] = data['dex_volume']
        
        logger.info("Fetched on-chain data")
        
    except Exception as e:
        logger.error(f"Error fetching on-chain data: {e}")
    
    return onchain_data


def save_ingested_data(data: Dict[str, pd.DataFrame], config: Dict) -> None:
    """
    Save ingested data to files.
    
    Args:
        data: Dictionary with ingested data
        config: Configuration dictionary
    """
    processed_dir = config['data']['processed_dir']
    
    for name, df in data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            try:
                save_data(df, name, processed_dir, file_format='parquet')
                logger.info(f"Saved {name} with shape {df.shape}")
            except Exception as e:
                logger.error(f"Error saving {name}: {e}")
        elif isinstance(df, pd.Series) and not df.empty:
            try:
                # Convert Series to DataFrame for saving
                df_to_save = df.to_frame()
                save_data(df_to_save, name, processed_dir, file_format='parquet')
                logger.info(f"Saved {name} with shape {df_to_save.shape}")
            except Exception as e:
                logger.error(f"Error saving {name}: {e}")


def main():
    """Main function for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run data ingestion pipeline')
    parser.add_argument('--config', default='configs/config.yaml', help='Configuration file path')
    parser.add_argument('--start-date', help='Start date for data ingestion')
    parser.add_argument('--end-date', help='End date for data ingestion')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    config = load_config(args.config)
    if args.start_date:
        config['analysis']['start_date'] = args.start_date
    if args.end_date:
        config['analysis']['end_date'] = args.end_date
    
    # Run pipeline
    try:
        ingested_data = run_data_ingestion(args.config)
        print(f"Successfully ingested {len(ingested_data)} datasets")
    except Exception as e:
        print(f"Error in data ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
