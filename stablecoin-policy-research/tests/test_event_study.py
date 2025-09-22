"""
Tests for event study analysis.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.event_study import EventStudyAnalyzer


class TestEventStudy:
    """Test event study analysis."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        returns = pd.DataFrame({
            'USDT': np.random.normal(0, 0.001, 300),
            'USDC': np.random.normal(0, 0.001, 300),
            'DAI': np.random.normal(0, 0.001, 300)
        }, index=dates)
        return returns
    
    @pytest.fixture
    def sample_market_returns(self):
        """Create sample market returns data."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        market_returns = pd.Series(np.random.normal(0, 0.01, 300), index=dates)
        return market_returns
    
    @pytest.fixture
    def sample_event_calendar(self):
        """Create sample event calendar."""
        events = pd.DataFrame({
            'event_date': pd.to_datetime(['2020-06-15', '2020-09-15', '2020-12-15']),
            'event_type': ['fomc_meeting', 'fomc_minutes', 'rate_decision'],
            'title': ['FOMC Meeting', 'FOMC Minutes', 'Rate Decision'],
            'event_window_start': pd.to_datetime(['2020-06-10', '2020-09-10', '2020-12-10']),
            'event_window_end': pd.to_datetime(['2020-06-20', '2020-09-20', '2020-12-20']),
            'estimation_window_start': pd.to_datetime(['2020-01-01', '2020-04-01', '2020-07-01']),
            'estimation_window_end': pd.to_datetime(['2020-06-09', '2020-09-09', '2020-12-09'])
        })
        return events
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            'policy_events': {
                'event_window': {'pre': 5, 'post': 5},
                'estimation_window': 250
            },
            'analysis': {
                'event_study': {'confidence_level': 0.95}
            }
        }
    
    @pytest.fixture
    def analyzer(self, config):
        """Create event study analyzer."""
        return EventStudyAnalyzer(config)
    
    def test_estimate_market_model(self, analyzer, sample_returns, sample_market_returns):
        """Test market model estimation."""
        asset_returns = sample_returns['USDT']
        
        alpha, beta = analyzer._estimate_market_model(asset_returns, sample_market_returns)
        
        # Check that parameters are reasonable
        assert isinstance(alpha, (int, float))
        assert isinstance(beta, (int, float))
        assert not np.isnan(alpha)
        assert not np.isnan(beta)
    
    def test_estimate_market_model_with_rf(self, analyzer, sample_returns, sample_market_returns):
        """Test market model estimation with risk-free rate."""
        asset_returns = sample_returns['USDT']
        risk_free_rate = pd.Series(0.001, index=sample_market_returns.index)
        
        alpha, beta = analyzer._estimate_market_model(asset_returns, sample_market_returns, risk_free_rate)
        
        # Check that parameters are reasonable
        assert isinstance(alpha, (int, float))
        assert isinstance(beta, (int, float))
        assert not np.isnan(alpha)
        assert not np.isnan(beta)
    
    def test_calculate_abnormal_returns(self, analyzer, sample_returns, sample_market_returns, sample_event_calendar):
        """Test abnormal returns calculation."""
        abnormal_returns = analyzer._calculate_abnormal_returns(
            sample_returns, sample_market_returns, sample_event_calendar
        )
        
        # Check that we get results for each stablecoin
        assert len(abnormal_returns) == len(sample_returns.columns)
        
        for stablecoin, ar_data in abnormal_returns.items():
            assert stablecoin in sample_returns.columns
            assert isinstance(ar_data, pd.DataFrame)
            
            if not ar_data.empty:
                # Check required columns
                required_columns = ['event_id', 'event_date', 'date', 'abnormal_return', 'alpha', 'beta']
                for col in required_columns:
                    assert col in ar_data.columns
    
    def test_calculate_car_windows(self, analyzer):
        """Test CAR window calculation."""
        # Create sample abnormal returns
        dates = pd.date_range('2020-06-10', periods=11, freq='D')
        ar_data = pd.DataFrame({
            'date': dates,
            'abnormal_return': np.random.normal(0, 0.001, 11),
            'event_date': pd.to_datetime('2020-06-15')
        })
        
        car_windows = analyzer._calculate_car_windows(ar_data)
        
        # Check that we get expected windows
        expected_windows = ['car_0_1', 'car_0_5', 'car_m1_1', 'car_m5_5']
        for window in expected_windows:
            assert window in car_windows
            assert isinstance(car_windows[window], (int, float))
    
    def test_calculate_car(self, analyzer, sample_returns, sample_market_returns, sample_event_calendar):
        """Test cumulative abnormal returns calculation."""
        abnormal_returns = analyzer._calculate_abnormal_returns(
            sample_returns, sample_market_returns, sample_event_calendar
        )
        
        car_results = analyzer._calculate_car(abnormal_returns, sample_event_calendar)
        
        # Check that we get results for each stablecoin
        assert len(car_results) == len(sample_returns.columns)
        
        for stablecoin, car_data in car_results.items():
            assert stablecoin in sample_returns.columns
            assert isinstance(car_data, pd.DataFrame)
            
            if not car_data.empty:
                # Check required columns
                required_columns = ['event_id', 'event_date']
                car_columns = ['car_0_1', 'car_0_5', 'car_m1_1', 'car_m5_5']
                
                for col in required_columns + car_columns:
                    assert col in car_data.columns
    
    def test_calculate_bhar(self, analyzer, sample_returns, sample_market_returns, sample_event_calendar):
        """Test buy-and-hold abnormal returns calculation."""
        bhar_results = analyzer._calculate_bhar(
            sample_returns, sample_market_returns, sample_event_calendar
        )
        
        # Check that we get results for each stablecoin
        assert len(bhar_results) == len(sample_returns.columns)
        
        for stablecoin, bhar_data in bhar_results.items():
            assert stablecoin in sample_returns.columns
            assert isinstance(bhar_data, pd.DataFrame)
            
            if not bhar_data.empty:
                # Check required columns
                required_columns = ['event_id', 'event_date']
                bhar_columns = ['bhar_1d', 'bhar_5d', 'bhar_10d', 'bhar_20d', 'bhar_60d']
                
                for col in required_columns + bhar_columns:
                    assert col in bhar_data.columns
    
    def test_run_statistical_tests(self, analyzer, sample_returns, sample_market_returns, sample_event_calendar):
        """Test statistical tests."""
        # Calculate CAR and BHAR first
        abnormal_returns = analyzer._calculate_abnormal_returns(
            sample_returns, sample_market_returns, sample_event_calendar
        )
        car_results = analyzer._calculate_car(abnormal_returns, sample_event_calendar)
        bhar_results = analyzer._calculate_bhar(
            sample_returns, sample_market_returns, sample_event_calendar
        )
        
        # Run statistical tests
        test_results = analyzer._run_statistical_tests(car_results, bhar_results)
        
        # Check that we get test results
        assert isinstance(test_results, dict)
        
        # Check CAR tests
        for stablecoin in sample_returns.columns:
            car_key = f'{stablecoin}_car'
            if car_key in test_results:
                car_tests = test_results[car_key]
                assert isinstance(car_tests, dict)
                
                for window, test_data in car_tests.items():
                    assert isinstance(test_data, dict)
                    required_keys = ['mean', 'std', 't_statistic', 't_pvalue', 'n_observations']
                    for key in required_keys:
                        assert key in test_data
    
    def test_calculate_summary_statistics(self, analyzer, sample_returns, sample_market_returns, sample_event_calendar):
        """Test summary statistics calculation."""
        # Calculate CAR and BHAR first
        abnormal_returns = analyzer._calculate_abnormal_returns(
            sample_returns, sample_market_returns, sample_event_calendar
        )
        car_results = analyzer._calculate_car(abnormal_returns, sample_event_calendar)
        bhar_results = analyzer._calculate_bhar(
            sample_returns, sample_market_returns, sample_event_calendar
        )
        
        # Calculate summary statistics
        summary = analyzer._calculate_summary_statistics(car_results, bhar_results)
        
        # Check structure
        assert 'car' in summary
        assert 'bhar' in summary
        
        # Check CAR summary
        car_summary = summary['car']
        for stablecoin in sample_returns.columns:
            if stablecoin in car_summary:
                stablecoin_summary = car_summary[stablecoin]
                assert isinstance(stablecoin_summary, dict)
    
    def test_run_event_study(self, analyzer, sample_returns, sample_market_returns, sample_event_calendar):
        """Test complete event study analysis."""
        results = analyzer.run_event_study(
            sample_returns, sample_market_returns, sample_event_calendar
        )
        
        # Check that we get all expected results
        expected_keys = ['abnormal_returns', 'car', 'bhar', 'statistical_tests', 'summary']
        for key in expected_keys:
            assert key in results
        
        # Check abnormal returns
        abnormal_returns = results['abnormal_returns']
        assert len(abnormal_returns) == len(sample_returns.columns)
        
        # Check CAR results
        car_results = results['car']
        assert len(car_results) == len(sample_returns.columns)
        
        # Check BHAR results
        bhar_results = results['bhar']
        assert len(bhar_results) == len(sample_returns.columns)
        
        # Check statistical tests
        statistical_tests = results['statistical_tests']
        assert isinstance(statistical_tests, dict)
        
        # Check summary
        summary = results['summary']
        assert isinstance(summary, dict)
        assert 'car' in summary
        assert 'bhar' in summary
    
    def test_analyze_by_sentiment(self, analyzer, sample_returns, sample_market_returns, sample_event_calendar):
        """Test sentiment-based analysis."""
        # Run event study first
        event_study_results = analyzer.run_event_study(
            sample_returns, sample_market_returns, sample_event_calendar
        )
        
        # Create sample sentiment data
        sentiment_data = sample_event_calendar.copy()
        sentiment_data['consensus_sentiment'] = ['hawkish', 'dovish', 'neutral']
        
        # Analyze by sentiment
        sentiment_analysis = analyzer.analyze_by_sentiment(event_study_results, sentiment_data)
        
        # Check that we get sentiment categories
        expected_sentiments = ['hawkish', 'dovish', 'neutral']
        for sentiment in expected_sentiments:
            if sentiment in sentiment_analysis:
                sentiment_results = sentiment_analysis[sentiment]
                assert 'n_events' in sentiment_results
                assert 'car_results' in sentiment_results
    
    def test_empty_data(self, analyzer):
        """Test handling of empty data."""
        empty_returns = pd.DataFrame()
        empty_market_returns = pd.Series()
        empty_events = pd.DataFrame()
        
        # Test with empty data
        results = analyzer.run_event_study(empty_returns, empty_market_returns, empty_events)
        
        # Should return empty results
        assert results['abnormal_returns'] == {}
        assert results['car'] == {}
        assert results['bhar'] == {}
        assert results['statistical_tests'] == {}
        assert results['summary'] == {}
