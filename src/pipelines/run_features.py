"""
Feature engineering pipeline for stablecoin policy research.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, ensure_directory
from utils.io import save_data, load_data
from features.build_events import EventBuilder
from features.market_metrics import MarketMetricsCalculator
from features.sentiment.build_sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)


def run_feature_engineering(config_path: str = "configs/config.yaml") -> Dict[str, pd.DataFrame]:
    """
    Run complete feature engineering pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with engineered features
    """
    logger.info("Starting feature engineering pipeline")
    
    # Load configuration
    config = load_config(config_path)
    
    # Ensure output directories exist
    ensure_directory(config['data']['interim_dir'])
    ensure_directory(config['data']['processed_dir'])
    
    # Load ingested data
    logger.info("Loading ingested data")
    ingested_data = load_ingested_data(config)
    
    if not ingested_data:
        logger.error("No ingested data found. Run data ingestion first.")
        return {}
    
    # Initialize feature engineering components
    event_builder = EventBuilder(config)
    metrics_calculator = MarketMetricsCalculator(config)
    sentiment_analyzer = SentimentAnalyzer(config)
    
    # Engineer features
    engineered_features = {}
    
    try:
        # 1. Build event calendar
        logger.info("Building event calendar")
        event_calendar = build_event_calendar(event_builder, ingested_data)
        engineered_features['event_calendar'] = event_calendar
        
        # 2. Calculate market metrics
        logger.info("Calculating market metrics")
        market_metrics = calculate_market_metrics(metrics_calculator, ingested_data)
        engineered_features.update(market_metrics)
        
        # 3. Analyze sentiment
        logger.info("Analyzing sentiment")
        sentiment_results = analyze_sentiment(sentiment_analyzer, ingested_data)
        engineered_features.update(sentiment_results)
        
        # 4. Create combined features
        logger.info("Creating combined features")
        combined_features = create_combined_features(engineered_features)
        engineered_features.update(combined_features)
        
        # Save engineered features
        logger.info("Saving engineered features")
        save_engineered_features(engineered_features, config)
        
        logger.info("Feature engineering pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {e}")
        raise
    
    return engineered_features


def load_ingested_data(config: Dict) -> Dict[str, pd.DataFrame]:
    """
    Load ingested data from files.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with loaded data
    """
    processed_dir = config['data']['processed_dir']
    ingested_data = {}
    
    # List of expected data files
    expected_files = [
        'stablecoin_prices',
        'stablecoin_volumes', 
        'stablecoin_market_caps',
        'policy_events',
        'fed_funds_rate',
        'treasury_rates',
        'inflation_data',
        'employment_data',
        'gdp_data',
        'financial_indicators'
    ]
    
    for filename in expected_files:
        try:
            data = load_data(filename, processed_dir)
            if not data.empty:
                ingested_data[filename] = data
                logger.info(f"Loaded {filename} with shape {data.shape}")
        except FileNotFoundError:
            logger.warning(f"File {filename} not found")
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
    
    return ingested_data


def build_event_calendar(event_builder: EventBuilder, ingested_data: Dict) -> pd.DataFrame:
    """
    Build event calendar from policy events.
    
    Args:
        event_builder: Event builder instance
        ingested_data: Dictionary with ingested data
        
    Returns:
        DataFrame with event calendar
    """
    policy_events = ingested_data.get('policy_events', pd.DataFrame())
    
    if policy_events.empty:
        logger.warning("No policy events found")
        return pd.DataFrame()
    
    # Build event calendar
    event_calendar = event_builder.build_event_calendar(policy_events)
    
    # Categorize events
    categorized_events = event_builder.categorize_events(event_calendar)
    
    # Filter by importance
    important_events = event_builder.filter_events_by_importance(categorized_events, min_importance=5.0)
    
    logger.info(f"Built event calendar with {len(important_events)} important events")
    
    return important_events


def calculate_market_metrics(
    metrics_calculator: MarketMetricsCalculator, 
    ingested_data: Dict
) -> Dict[str, pd.DataFrame]:
    """
    Calculate market metrics from price data.
    
    Args:
        metrics_calculator: Market metrics calculator instance
        ingested_data: Dictionary with ingested data
        
    Returns:
        Dictionary with market metrics
    """
    market_metrics = {}
    
    # Get price data
    prices = ingested_data.get('stablecoin_prices', pd.DataFrame())
    volumes = ingested_data.get('stablecoin_volumes', pd.DataFrame())
    
    if prices.empty:
        logger.warning("No price data found")
        return market_metrics
    
    # Calculate all metrics
    all_metrics = metrics_calculator.calculate_all_metrics(prices, volumes)
    
    # Organize metrics
    market_metrics['returns'] = all_metrics.get('returns', pd.DataFrame())
    market_metrics['volatility'] = all_metrics.get('volatility', pd.DataFrame())
    market_metrics['peg_deviations'] = all_metrics.get('peg_deviations', pd.DataFrame())
    market_metrics['peg_deviation_probability'] = all_metrics.get('peg_deviation_probability', pd.DataFrame())
    market_metrics['rolling_correlation'] = all_metrics.get('rolling_correlation', pd.DataFrame())
    market_metrics['volatility_regime'] = all_metrics.get('volatility_regime', pd.DataFrame())
    
    if volumes is not None and not volumes.empty:
        market_metrics['market_depth'] = all_metrics.get('market_depth', pd.DataFrame())
        market_metrics['amihud'] = all_metrics.get('amihud', pd.DataFrame())
    
    logger.info(f"Calculated {len(market_metrics)} market metrics")
    
    return market_metrics


def analyze_sentiment(
    sentiment_analyzer: SentimentAnalyzer, 
    ingested_data: Dict
) -> Dict[str, pd.DataFrame]:
    """
    Analyze sentiment of policy events.
    
    Args:
        sentiment_analyzer: Sentiment analyzer instance
        ingested_data: Dictionary with ingested data
        
    Returns:
        Dictionary with sentiment analysis results
    """
    sentiment_results = {}
    
    # Get policy events
    policy_events = ingested_data.get('policy_events', pd.DataFrame())
    
    if policy_events.empty:
        logger.warning("No policy events found for sentiment analysis")
        return sentiment_results
    
    # Perform comprehensive sentiment analysis
    sentiment_analyzed_events = sentiment_analyzer.analyze_sentiment_comprehensive(
        policy_events, 
        text_column='content',
        methods=['both']
    )
    
    sentiment_results['policy_events_with_sentiment'] = sentiment_analyzed_events
    
    # Create sentiment time series
    sentiment_timeseries = sentiment_analyzer.analyze_sentiment_timeseries(
        sentiment_analyzed_events, freq='D'
    )
    
    sentiment_results['sentiment_timeseries'] = sentiment_timeseries
    
    # Get sentiment summary
    sentiment_summary = sentiment_analyzer.get_sentiment_summary(sentiment_analyzed_events)
    sentiment_results['sentiment_summary'] = pd.DataFrame([sentiment_summary])
    
    logger.info(f"Analyzed sentiment for {len(sentiment_analyzed_events)} events")
    
    return sentiment_results


def create_combined_features(engineered_features: Dict) -> Dict[str, pd.DataFrame]:
    """
    Create combined features from individual metrics.
    
    Args:
        engineered_features: Dictionary with engineered features
        
    Returns:
        Dictionary with combined features
    """
    combined_features = {}
    
    try:
        # Combine market metrics with sentiment
        market_metrics = engineered_features.get('returns', pd.DataFrame())
        sentiment_ts = engineered_features.get('sentiment_timeseries', pd.DataFrame())
        
        if not market_metrics.empty and not sentiment_ts.empty:
            # Align dates
            common_dates = market_metrics.index.intersection(sentiment_ts.index)
            
            if len(common_dates) > 0:
                combined_data = pd.DataFrame(index=common_dates)
                
                # Add market metrics
                for col in market_metrics.columns:
                    combined_data[col] = market_metrics.loc[common_dates, col]
                
                # Add sentiment metrics
                for col in sentiment_ts.columns:
                    combined_data[f'sentiment_{col}'] = sentiment_ts.loc[common_dates, col]
                
                combined_features['market_sentiment_combined'] = combined_data
                logger.info(f"Created combined features with {len(combined_data)} observations")
        
        # Create event-based features
        event_calendar = engineered_features.get('event_calendar', pd.DataFrame())
        returns = engineered_features.get('returns', pd.DataFrame())
        
        if not event_calendar.empty and not returns.empty:
            event_features = create_event_features(event_calendar, returns)
            combined_features.update(event_features)
        
    except Exception as e:
        logger.error(f"Error creating combined features: {e}")
    
    return combined_features


def create_event_features(
    event_calendar: pd.DataFrame, 
    returns: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Create features based on events.
    
    Args:
        event_calendar: DataFrame with event calendar
        returns: DataFrame with returns data
        
    Returns:
        Dictionary with event-based features
    """
    event_features = {}
    
    try:
        # Create event dummy variables
        event_dummies = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for _, event in event_calendar.iterrows():
            event_date = event['event_date']
            event_window_start = event['event_window_start']
            event_window_end = event['event_window_end']
            
            # Create dummy for event window
            event_window_dates = pd.date_range(event_window_start, event_window_end, freq='D')
            event_window_dates = event_window_dates[event_window_dates.isin(returns.index)]
            
            if len(event_window_dates) > 0:
                for col in returns.columns:
                    event_dummies.loc[event_window_dates, col] = 1
        
        # Fill NaN with 0
        event_dummies = event_dummies.fillna(0)
        
        event_features['event_dummies'] = event_dummies
        
        # Create sentiment-based event features
        if 'consensus_sentiment' in event_calendar.columns:
            sentiment_event_features = create_sentiment_event_features(event_calendar, returns)
            event_features.update(sentiment_event_features)
        
    except Exception as e:
        logger.error(f"Error creating event features: {e}")
    
    return event_features


def create_sentiment_event_features(
    event_calendar: pd.DataFrame, 
    returns: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Create features based on sentiment-categorized events.
    
    Args:
        event_calendar: DataFrame with event calendar
        returns: DataFrame with returns data
        
    Returns:
        Dictionary with sentiment event features
    """
    sentiment_features = {}
    
    try:
        sentiment_categories = ['hawkish', 'dovish', 'neutral']
        
        for sentiment in sentiment_categories:
            # Filter events by sentiment
            sentiment_events = event_calendar[event_calendar['consensus_sentiment'] == sentiment]
            
            if len(sentiment_events) == 0:
                continue
            
            # Create dummy variables for sentiment events
            sentiment_dummies = pd.DataFrame(index=returns.index, columns=returns.columns)
            
            for _, event in sentiment_events.iterrows():
                event_window_start = event['event_window_start']
                event_window_end = event['event_window_end']
                
                event_window_dates = pd.date_range(event_window_start, event_window_end, freq='D')
                event_window_dates = event_window_dates[event_window_dates.isin(returns.index)]
                
                if len(event_window_dates) > 0:
                    for col in returns.columns:
                        sentiment_dummies.loc[event_window_dates, col] = 1
            
            # Fill NaN with 0
            sentiment_dummies = sentiment_dummies.fillna(0)
            
            sentiment_features[f'{sentiment}_event_dummies'] = sentiment_dummies
        
    except Exception as e:
        logger.error(f"Error creating sentiment event features: {e}")
    
    return sentiment_features


def save_engineered_features(features: Dict[str, pd.DataFrame], config: Dict) -> None:
    """
    Save engineered features to files.
    
    Args:
        features: Dictionary with engineered features
        config: Configuration dictionary
    """
    processed_dir = config['data']['processed_dir']
    
    for name, df in features.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            try:
                save_data(df, name, processed_dir, file_format='parquet')
                logger.info(f"Saved {name} with shape {df.shape}")
            except Exception as e:
                logger.error(f"Error saving {name}: {e}")


def main():
    """Main function for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run feature engineering pipeline')
    parser.add_argument('--config', default='configs/config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Run pipeline
    try:
        engineered_features = run_feature_engineering(args.config)
        print(f"Successfully engineered {len(engineered_features)} feature sets")
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
