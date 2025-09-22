"""
Market metrics calculator for stablecoin analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class MarketMetricsCalculator:
    """
    Calculates various market metrics for stablecoin analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize market metrics calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.peg_threshold = config.get('market_metrics', {}).get('peg_deviation', {}).get('threshold', 0.01)
        self.volatility_window = config.get('market_metrics', {}).get('volatility', {}).get('window', 30)
    
    def calculate_returns(
        self, 
        prices: pd.DataFrame, 
        method: str = 'log'
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            prices: DataFrame with price data
            method: Method for calculating returns ('log' or 'simple')
            
        Returns:
            DataFrame with returns
        """
        logger.info(f"Calculating {method} returns")
        
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        elif method == 'simple':
            returns = (prices / prices.shift(1)) - 1
        else:
            raise ValueError(f"Unknown return method: {method}")
        
        # Remove first row (NaN)
        returns = returns.dropna()
        
        return returns
    
    def calculate_volatility(
        self, 
        returns: pd.DataFrame, 
        window: int = None,
        method: str = 'rolling'
    ) -> pd.DataFrame:
        """
        Calculate volatility from returns.
        
        Args:
            returns: DataFrame with returns data
            window: Rolling window size
            method: Method for calculating volatility ('rolling', 'ewm')
            
        Returns:
            DataFrame with volatility
        """
        if window is None:
            window = self.volatility_window
        
        logger.info(f"Calculating {method} volatility with window {window}")
        
        if method == 'rolling':
            volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        elif method == 'ewm':
            volatility = returns.ewm(span=window).std() * np.sqrt(252)  # Annualized
        else:
            raise ValueError(f"Unknown volatility method: {method}")
        
        return volatility
    
    def calculate_peg_deviations(
        self, 
        prices: pd.DataFrame, 
        peg_price: float = 1.0
    ) -> pd.DataFrame:
        """
        Calculate peg deviations from target price.
        
        Args:
            prices: DataFrame with price data
            peg_price: Target peg price (default 1.0 for USD-pegged stablecoins)
            
        Returns:
            DataFrame with peg deviations
        """
        logger.info(f"Calculating peg deviations from {peg_price}")
        
        deviations = prices - peg_price
        
        return deviations
    
    def calculate_peg_deviation_probability(
        self, 
        peg_deviations: pd.DataFrame, 
        threshold: float = None
    ) -> pd.DataFrame:
        """
        Calculate probability of exceeding peg deviation threshold.
        
        Args:
            peg_deviations: DataFrame with peg deviations
            threshold: Deviation threshold (default from config)
            
        Returns:
            DataFrame with probabilities
        """
        if threshold is None:
            threshold = self.peg_threshold
        
        logger.info(f"Calculating peg deviation probabilities for threshold {threshold}")
        
        # Calculate probability of exceeding threshold
        probabilities = pd.DataFrame(index=peg_deviations.index, columns=peg_deviations.columns)
        
        for col in peg_deviations.columns:
            # Rolling probability calculation
            for i in range(len(peg_deviations)):
                if i < 30:  # Need minimum observations
                    probabilities.loc[peg_deviations.index[i], col] = np.nan
                else:
                    window_data = peg_deviations[col].iloc[:i+1].dropna()
                    if len(window_data) > 0:
                        prob = np.mean(np.abs(window_data) > threshold)
                        probabilities.loc[peg_deviations.index[i], col] = prob
        
        return probabilities
    
    def calculate_market_depth(
        self, 
        volumes: pd.DataFrame, 
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate market depth metrics.
        
        Args:
            volumes: DataFrame with volume data
            prices: DataFrame with price data
            
        Returns:
            DataFrame with market depth metrics
        """
        logger.info("Calculating market depth metrics")
        
        # Market depth as volume-weighted average price impact
        depth_metrics = pd.DataFrame(index=volumes.index, columns=volumes.columns)
        
        for col in volumes.columns:
            if col in prices.columns:
                # Calculate price impact proxy
                price_changes = prices[col].pct_change().abs()
                volume_weighted_impact = (price_changes / volumes[col]).rolling(7).mean()
                depth_metrics[col] = 1 / (volume_weighted_impact + 1e-8)  # Inverse of impact
        
        return depth_metrics
    
    def calculate_liquidity_metrics(
        self, 
        prices: pd.DataFrame, 
        volumes: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate various liquidity metrics.
        
        Args:
            prices: DataFrame with price data
            volumes: DataFrame with volume data
            
        Returns:
            Dictionary with liquidity metrics
        """
        logger.info("Calculating liquidity metrics")
        
        liquidity_metrics = {}
        
        # Bid-ask spread proxy (using intraday volatility)
        returns = self.calculate_returns(prices)
        volatility = self.calculate_volatility(returns)
        liquidity_metrics['spread_proxy'] = volatility
        
        # Volume-weighted average price (VWAP) deviation
        vwap_deviation = pd.DataFrame(index=prices.index, columns=prices.columns)
        for col in prices.columns:
            if col in volumes.columns:
                vwap = (prices[col] * volumes[col]).rolling(24).sum() / volumes[col].rolling(24).sum()
                vwap_deviation[col] = (prices[col] - vwap) / vwap
        
        liquidity_metrics['vwap_deviation'] = vwap_deviation
        
        # Amihud illiquidity measure (price impact per unit volume)
        amihud = pd.DataFrame(index=prices.index, columns=prices.columns)
        for col in prices.columns:
            if col in volumes.columns:
                returns_col = returns[col].abs()
                volumes_col = volumes[col]
                amihud[col] = (returns_col / volumes_col).rolling(7).mean()
        
        liquidity_metrics['amihud'] = amihud
        
        return liquidity_metrics
    
    def calculate_correlation_metrics(
        self, 
        returns: pd.DataFrame, 
        window: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate correlation metrics between stablecoins.
        
        Args:
            returns: DataFrame with returns data
            window: Rolling window for correlation calculation
            
        Returns:
            Dictionary with correlation metrics
        """
        logger.info(f"Calculating correlation metrics with window {window}")
        
        correlation_metrics = {}
        
        # Rolling correlation matrix
        rolling_corr = returns.rolling(window=window).corr()
        correlation_metrics['rolling_correlation'] = rolling_corr
        
        # Average correlation over time
        avg_corr = rolling_corr.groupby(level=0).mean()
        correlation_metrics['average_correlation'] = avg_corr
        
        # Correlation stability (standard deviation of rolling correlations)
        corr_stability = rolling_corr.groupby(level=0).std()
        correlation_metrics['correlation_stability'] = corr_stability
        
        return correlation_metrics
    
    def calculate_regime_metrics(
        self, 
        returns: pd.DataFrame, 
        window: int = 60
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate regime change metrics.
        
        Args:
            returns: DataFrame with returns data
            window: Window for regime detection
            
        Returns:
            Dictionary with regime metrics
        """
        logger.info(f"Calculating regime metrics with window {window}")
        
        regime_metrics = {}
        
        # Volatility regimes
        volatility = self.calculate_volatility(returns)
        vol_regimes = pd.DataFrame(index=volatility.index, columns=volatility.columns)
        
        for col in volatility.columns:
            vol_series = volatility[col].dropna()
            if len(vol_series) > window:
                # Simple regime detection based on volatility percentiles
                vol_threshold = vol_series.rolling(window).quantile(0.8)
                vol_regimes[col] = (vol_series > vol_threshold).astype(int)
        
        regime_metrics['volatility_regime'] = vol_regimes
        
        # Return regime (positive/negative)
        return_regimes = (returns > 0).astype(int)
        regime_metrics['return_regime'] = return_regimes
        
        return regime_metrics
    
    def calculate_all_metrics(
        self, 
        prices: pd.DataFrame, 
        volumes: Optional[pd.DataFrame] = None
    ) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        Calculate all market metrics.
        
        Args:
            prices: DataFrame with price data
            volumes: Optional DataFrame with volume data
            
        Returns:
            Dictionary with all calculated metrics
        """
        logger.info("Calculating all market metrics")
        
        all_metrics = {}
        
        # Basic metrics
        returns = self.calculate_returns(prices)
        all_metrics['returns'] = returns
        
        volatility = self.calculate_volatility(returns)
        all_metrics['volatility'] = volatility
        
        peg_deviations = self.calculate_peg_deviations(prices)
        all_metrics['peg_deviations'] = peg_deviations
        
        peg_probabilities = self.calculate_peg_deviation_probability(peg_deviations)
        all_metrics['peg_deviation_probability'] = peg_probabilities
        
        # Correlation metrics
        correlation_metrics = self.calculate_correlation_metrics(returns)
        all_metrics.update(correlation_metrics)
        
        # Regime metrics
        regime_metrics = self.calculate_regime_metrics(returns)
        all_metrics.update(regime_metrics)
        
        # Volume-based metrics (if available)
        if volumes is not None:
            market_depth = self.calculate_market_depth(volumes, prices)
            all_metrics['market_depth'] = market_depth
            
            liquidity_metrics = self.calculate_liquidity_metrics(prices, volumes)
            all_metrics.update(liquidity_metrics)
        
        return all_metrics
    
    def get_metric_summary(self, metrics: Dict[str, pd.DataFrame]) -> Dict:
        """
        Get summary statistics for all metrics.
        
        Args:
            metrics: Dictionary with calculated metrics
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, pd.DataFrame):
                summary[metric_name] = {
                    'mean': metric_data.mean().mean(),
                    'std': metric_data.std().mean(),
                    'min': metric_data.min().min(),
                    'max': metric_data.max().max(),
                    'shape': metric_data.shape
                }
        
        return summary
