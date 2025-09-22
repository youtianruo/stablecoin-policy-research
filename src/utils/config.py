"""
Configuration management utilities.
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file and environment variables.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    # Load environment variables
    load_dotenv()
    
    # Load YAML configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Override with environment variables where applicable
    config = _override_with_env_vars(config)
    
    return config


def _override_with_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override configuration values with environment variables.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Updated configuration dictionary
    """
    # API keys
    if 'OPENAI_API_KEY' in os.environ:
        config.setdefault('api_keys', {})['openai'] = os.environ['OPENAI_API_KEY']
    
    if 'COINGECKO_API_KEY' in os.environ:
        config.setdefault('api_keys', {})['coingecko'] = os.environ['COINGECKO_API_KEY']
    
    if 'FRED_API_KEY' in os.environ:
        config.setdefault('api_keys', {})['fred'] = os.environ['FRED_API_KEY']
    
    if 'ETHERSCAN_API_KEY' in os.environ:
        config.setdefault('api_keys', {})['etherscan'] = os.environ['ETHERSCAN_API_KEY']
    
    if 'DUNE_API_KEY' in os.environ:
        config.setdefault('api_keys', {})['dune'] = os.environ['DUNE_API_KEY']
    
    # Analysis parameters
    if 'DEFAULT_START_DATE' in os.environ:
        config.setdefault('analysis', {})['start_date'] = os.environ['DEFAULT_START_DATE']
    
    if 'DEFAULT_END_DATE' in os.environ:
        config.setdefault('analysis', {})['end_date'] = os.environ['DEFAULT_END_DATE']
    
    if 'STABLECOIN_TICKERS' in os.environ:
        tickers = os.environ['STABLECOIN_TICKERS'].split(',')
        config.setdefault('stablecoins', {})['tickers'] = [t.strip() for t in tickers]
    
    # Event study parameters
    if 'EVENT_WINDOW_PRE' in os.environ:
        config.setdefault('policy_events', {}).setdefault('event_window', {})['pre'] = int(os.environ['EVENT_WINDOW_PRE'])
    
    if 'EVENT_WINDOW_POST' in os.environ:
        config.setdefault('policy_events', {}).setdefault('event_window', {})['post'] = int(os.environ['EVENT_WINDOW_POST'])
    
    if 'ESTIMATION_WINDOW' in os.environ:
        config.setdefault('policy_events', {})['estimation_window'] = int(os.environ['ESTIMATION_WINDOW'])
    
    # Sentiment analysis
    if 'SENTIMENT_MODEL' in os.environ:
        config.setdefault('sentiment', {})['model'] = os.environ['SENTIMENT_MODEL']
    
    # Output directories
    if 'DATA_DIR' in os.environ:
        config.setdefault('data', {})['raw_dir'] = os.path.join(os.environ['DATA_DIR'], 'raw')
        config.setdefault('data', {})['interim_dir'] = os.path.join(os.environ['DATA_DIR'], 'interim')
        config.setdefault('data', {})['processed_dir'] = os.path.join(os.environ['DATA_DIR'], 'processed')
    
    if 'OUTPUT_DIR' in os.environ:
        config.setdefault('output', {})['results_dir'] = os.environ['OUTPUT_DIR']
    
    if 'FIGURES_DIR' in os.environ:
        config.setdefault('output', {})['figures_dir'] = os.environ['FIGURES_DIR']
    
    # Logging
    if 'LOG_LEVEL' in os.environ:
        config.setdefault('logging', {})['level'] = os.environ['LOG_LEVEL']
    
    return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the configuration value
        default: Default value if key is not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def ensure_directory(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
    """
    Path(path).mkdir(parents=True, exist_ok=True)
