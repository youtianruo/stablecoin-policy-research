"""
Tests for market metrics calculation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.market_metrics import MarketMetricsCalculator


class TestMarketMetrics:
    """Test market metrics calculation."""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = pd.DataFrame({
            'USDT': 1.0 + np.random.normal(0, 0.001, 100),
            'USDC': 1.0 + np.random.normal(0, 0.001, 100),
            'DAI': 1.0 + np.random.normal(0, 0.001, 100)
        }, index=dates)
        return prices
    
    @pytest.fixture
    def sample_volumes(self):
        """Create sample volume data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        volumes = pd.DataFrame({
            'USDT': np.random.exponential(1000000, 100),
            'USDC': np.random.exponential(500000, 100),
            'DAI': np.random.exponential(200000, 100)
        }, index=dates)
        return volumes
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            'market_metrics': {
                'peg_deviation': {'threshold': 0.01},
                'volatility': {'window': 30}
            }
        }
    
    @pytest.fixture
    def calculator(self, config):
        """Create market metrics calculator."""
        return MarketMetricsCalculator(config)
    
    def test_calculate_returns_log(self, calculator, sample_prices):
        """Test log returns calculation."""
        returns = calculator.calculate_returns(sample_prices, method='log')
        
        # Check shape
        assert returns.shape == sample_prices.shape
        
        # Check that first row is NaN
        assert returns.iloc[0].isna().all()
        
        # Check that returns are approximately log returns
        expected_log_returns = np.log(sample_prices / sample_prices.shift(1))
        pd.testing.assert_frame_equal(returns, expected_log_returns)
    
    def test_calculate_returns_simple(self, calculator, sample_prices):
        """Test simple returns calculation."""
        returns = calculator.calculate_returns(sample_prices, method='simple')
        
        # Check shape
        assert returns.shape == sample_prices.shape
        
        # Check that first row is NaN
        assert returns.iloc[0].isna().all()
        
        # Check that returns are approximately simple returns
        expected_simple_returns = (sample_prices / sample_prices.shift(1)) - 1
        pd.testing.assert_frame_equal(returns, expected_simple_returns)
    
    def test_calculate_volatility(self, calculator, sample_prices):
        """Test volatility calculation."""
        returns = calculator.calculate_returns(sample_prices, method='log')
        volatility = calculator.calculate_volatility(returns, window=30)
        
        # Check shape
        assert volatility.shape == returns.shape
        
        # Check that volatility is positive
        assert (volatility >= 0).all().all()
        
        # Check that volatility is annualized (should be reasonable values)
        assert volatility.max().max() < 1.0  # Less than 100% annualized volatility
    
    def test_calculate_peg_deviations(self, calculator, sample_prices):
        """Test peg deviation calculation."""
        deviations = calculator.calculate_peg_deviations(sample_prices, peg_price=1.0)
        
        # Check shape
        assert deviations.shape == sample_prices.shape
        
        # Check that deviations are calculated correctly
        expected_deviations = sample_prices - 1.0
        pd.testing.assert_frame_equal(deviations, expected_deviations)
    
    def test_calculate_peg_deviation_probability(self, calculator, sample_prices):
        """Test peg deviation probability calculation."""
        deviations = calculator.calculate_peg_deviations(sample_prices, peg_price=1.0)
        probabilities = calculator.calculate_peg_deviation_probability(deviations, threshold=0.01)
        
        # Check shape
        assert probabilities.shape == deviations.shape
        
        # Check that probabilities are between 0 and 1
        assert (probabilities >= 0).all().all()
        assert (probabilities <= 1).all().all()
    
    def test_calculate_market_depth(self, calculator, sample_prices, sample_volumes):
        """Test market depth calculation."""
        depth = calculator.calculate_market_depth(sample_volumes, sample_prices)
        
        # Check shape
        assert depth.shape == sample_volumes.shape
        
        # Check that depth values are positive
        assert (depth >= 0).all().all()
    
    def test_calculate_liquidity_metrics(self, calculator, sample_prices, sample_volumes):
        """Test liquidity metrics calculation."""
        liquidity_metrics = calculator.calculate_liquidity_metrics(sample_prices, sample_volumes)
        
        # Check that we get expected metrics
        expected_metrics = ['spread_proxy', 'vwap_deviation', 'amihud']
        for metric in expected_metrics:
            assert metric in liquidity_metrics
        
        # Check shapes
        for metric_name, metric_data in liquidity_metrics.items():
            assert metric_data.shape == sample_prices.shape
    
    def test_calculate_correlation_metrics(self, calculator, sample_prices):
        """Test correlation metrics calculation."""
        returns = calculator.calculate_returns(sample_prices, method='log')
        correlation_metrics = calculator.calculate_correlation_metrics(returns, window=30)
        
        # Check that we get expected metrics
        expected_metrics = ['rolling_correlation', 'average_correlation', 'correlation_stability']
        for metric in expected_metrics:
            assert metric in correlation_metrics
        
        # Check that correlations are between -1 and 1
        rolling_corr = correlation_metrics['rolling_correlation']
        if not rolling_corr.empty:
            assert (rolling_corr >= -1).all().all()
            assert (rolling_corr <= 1).all().all()
    
    def test_calculate_regime_metrics(self, calculator, sample_prices):
        """Test regime metrics calculation."""
        returns = calculator.calculate_returns(sample_prices, method='log')
        regime_metrics = calculator.calculate_regime_metrics(returns, window=60)
        
        # Check that we get expected metrics
        expected_metrics = ['volatility_regime', 'return_regime']
        for metric in expected_metrics:
            assert metric in regime_metrics
        
        # Check that regime indicators are binary
        for metric_name, metric_data in regime_metrics.items():
            if not metric_data.empty:
                unique_values = metric_data.dropna().unique()
                assert all(val in [0, 1] for val in unique_values)
    
    def test_calculate_all_metrics(self, calculator, sample_prices, sample_volumes):
        """Test calculation of all metrics."""
        all_metrics = calculator.calculate_all_metrics(sample_prices, sample_volumes)
        
        # Check that we get expected metrics
        expected_metrics = [
            'returns', 'volatility', 'peg_deviations', 'peg_deviation_probability',
            'rolling_correlation', 'volatility_regime', 'market_depth', 'amihud'
        ]
        
        for metric in expected_metrics:
            assert metric in all_metrics
        
        # Check shapes
        for metric_name, metric_data in all_metrics.items():
            if isinstance(metric_data, pd.DataFrame):
                assert metric_data.shape[0] == sample_prices.shape[0]  # Same number of rows
    
    def test_get_metric_summary(self, calculator, sample_prices, sample_volumes):
        """Test metric summary generation."""
        all_metrics = calculator.calculate_all_metrics(sample_prices, sample_volumes)
        summary = calculator.get_metric_summary(all_metrics)
        
        # Check that summary contains expected keys
        for metric_name in all_metrics.keys():
            assert metric_name in summary
        
        # Check that summary values are reasonable
        for metric_name, metric_summary in summary.items():
            assert 'mean' in metric_summary
            assert 'std' in metric_summary
            assert 'shape' in metric_summary
            assert isinstance(metric_summary['mean'], (int, float))
            assert isinstance(metric_summary['std'], (int, float))
    
    def test_invalid_return_method(self, calculator, sample_prices):
        """Test error handling for invalid return method."""
        with pytest.raises(ValueError):
            calculator.calculate_returns(sample_prices, method='invalid')
    
    def test_invalid_volatility_method(self, calculator, sample_prices):
        """Test error handling for invalid volatility method."""
        returns = calculator.calculate_returns(sample_prices, method='log')
        with pytest.raises(ValueError):
            calculator.calculate_volatility(returns, method='invalid')
    
    def test_empty_dataframe(self, calculator):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        
        # Test returns calculation
        returns = calculator.calculate_returns(empty_df)
        assert returns.empty
        
        # Test volatility calculation
        volatility = calculator.calculate_volatility(empty_df)
        assert volatility.empty
        
        # Test peg deviations
        deviations = calculator.calculate_peg_deviations(empty_df)
        assert deviations.empty
