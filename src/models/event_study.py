"""
Event study analysis for stablecoin market behavior around policy events.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EventStudyAnalyzer:
    """
    Performs event study analysis for stablecoin markets.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize event study analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.event_window_pre = config.get('policy_events', {}).get('event_window', {}).get('pre', 5)
        self.event_window_post = config.get('policy_events', {}).get('event_window', {}).get('post', 5)
        self.estimation_window = config.get('policy_events', {}).get('estimation_window', 250)
        self.confidence_level = config.get('analysis', {}).get('event_study', {}).get('confidence_level', 0.95)
    
    def run_event_study(
        self, 
        returns: pd.DataFrame,
        market_returns: pd.Series,
        event_calendar: pd.DataFrame,
        risk_free_rate: Optional[pd.Series] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run comprehensive event study analysis.
        
        Args:
            returns: DataFrame with stablecoin returns
            market_returns: Series with market returns (benchmark)
            event_calendar: DataFrame with event calendar
            risk_free_rate: Optional Series with risk-free rate
            
        Returns:
            Dictionary with event study results
        """
        logger.info("Running event study analysis")
        
        results = {}
        
        # Calculate abnormal returns
        abnormal_returns = self._calculate_abnormal_returns(
            returns, market_returns, event_calendar, risk_free_rate
        )
        results['abnormal_returns'] = abnormal_returns
        
        # Calculate cumulative abnormal returns (CAR)
        car_results = self._calculate_car(abnormal_returns, event_calendar)
        results['car'] = car_results
        
        # Calculate buy-and-hold abnormal returns (BHAR)
        bhar_results = self._calculate_bhar(returns, market_returns, event_calendar, risk_free_rate)
        results['bhar'] = bhar_results
        
        # Statistical tests
        statistical_tests = self._run_statistical_tests(car_results, bhar_results)
        results['statistical_tests'] = statistical_tests
        
        # Summary statistics
        summary = self._calculate_summary_statistics(car_results, bhar_results)
        results['summary'] = summary
        
        return results
    
    def _calculate_abnormal_returns(
        self, 
        returns: pd.DataFrame,
        market_returns: pd.Series,
        event_calendar: pd.DataFrame,
        risk_free_rate: Optional[pd.Series] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate abnormal returns using market model.
        
        Args:
            returns: DataFrame with stablecoin returns
            market_returns: Series with market returns
            event_calendar: DataFrame with event calendar
            risk_free_rate: Optional Series with risk-free rate
            
        Returns:
            Dictionary with abnormal returns for each stablecoin
        """
        logger.info("Calculating abnormal returns")
        
        abnormal_returns = {}
        
        for stablecoin in returns.columns:
            logger.info(f"Calculating abnormal returns for {stablecoin}")
            
            ar_data = []
            
            for _, event in event_calendar.iterrows():
                event_date = event['event_date']
                estimation_start = event['estimation_window_start']
                estimation_end = event['estimation_window_end']
                event_start = event['event_window_start']
                event_end = event['event_window_end']
                
                # Get estimation period data
                est_returns = returns[stablecoin].loc[estimation_start:estimation_end]
                est_market = market_returns.loc[estimation_start:estimation_end]
                
                if len(est_returns) < 30:  # Need minimum observations
                    continue
                
                # Calculate market model parameters
                alpha, beta = self._estimate_market_model(est_returns, est_market, risk_free_rate)
                
                # Get event window data
                event_returns = returns[stablecoin].loc[event_start:event_end]
                event_market = market_returns.loc[event_start:event_end]
                
                if len(event_returns) == 0:
                    continue
                
                # Calculate expected returns
                if risk_free_rate is not None:
                    event_rf = risk_free_rate.loc[event_start:event_end]
                    expected_returns = alpha + beta * (event_market - event_rf)
                else:
                    expected_returns = alpha + beta * event_market
                
                # Calculate abnormal returns
                abnormal_ret = event_returns - expected_returns
                
                # Store results
                for date, ar in abnormal_ret.items():
                    ar_data.append({
                        'event_id': event.name,
                        'event_date': event_date,
                        'date': date,
                        'abnormal_return': ar,
                        'alpha': alpha,
                        'beta': beta,
                        'expected_return': expected_returns.get(date, np.nan),
                        'actual_return': event_returns.get(date, np.nan)
                    })
            
            abnormal_returns[stablecoin] = pd.DataFrame(ar_data)
        
        return abnormal_returns
    
    def _estimate_market_model(
        self, 
        returns: pd.Series, 
        market_returns: pd.Series,
        risk_free_rate: Optional[pd.Series] = None
    ) -> Tuple[float, float]:
        """
        Estimate market model parameters (alpha, beta).
        
        Args:
            returns: Asset returns
            market_returns: Market returns
            risk_free_rate: Optional risk-free rate
            
        Returns:
            Tuple of (alpha, beta)
        """
        # Align data
        common_dates = returns.index.intersection(market_returns.index)
        if len(common_dates) < 10:
            return 0.0, 1.0
        
        asset_ret = returns.loc[common_dates]
        market_ret = market_returns.loc[common_dates]
        
        # Calculate excess returns if risk-free rate available
        if risk_free_rate is not None:
            rf_ret = risk_free_rate.loc[common_dates]
            asset_excess = asset_ret - rf_ret
            market_excess = market_ret - rf_ret
        else:
            asset_excess = asset_ret
            market_excess = market_ret
        
        # Remove NaN values
        valid_data = pd.DataFrame({
            'asset': asset_excess,
            'market': market_excess
        }).dropna()
        
        if len(valid_data) < 10:
            return 0.0, 1.0
        
        # Estimate market model
        X = valid_data['market'].values.reshape(-1, 1)
        y = valid_data['asset'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        alpha = model.intercept_
        beta = model.coef_[0]
        
        return alpha, beta
    
    def _calculate_car(
        self, 
        abnormal_returns: Dict[str, pd.DataFrame],
        event_calendar: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate cumulative abnormal returns (CAR).
        
        Args:
            abnormal_returns: Dictionary with abnormal returns
            event_calendar: DataFrame with event calendar
            
        Returns:
            Dictionary with CAR results
        """
        logger.info("Calculating cumulative abnormal returns")
        
        car_results = {}
        
        for stablecoin, ar_data in abnormal_returns.items():
            if ar_data.empty:
                continue
            
            car_data = []
            
            for event_id in ar_data['event_id'].unique():
                event_ar = ar_data[ar_data['event_id'] == event_id].sort_values('date')
                
                if len(event_ar) == 0:
                    continue
                
                # Calculate CAR for different windows
                car_windows = self._calculate_car_windows(event_ar)
                
                car_data.append({
                    'event_id': event_id,
                    'event_date': event_ar['event_date'].iloc[0],
                    **car_windows
                })
            
            car_results[stablecoin] = pd.DataFrame(car_data)
        
        return car_results
    
    def _calculate_car_windows(self, event_ar: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate CAR for different event windows.
        
        Args:
            event_ar: Abnormal returns for a single event
            
        Returns:
            Dictionary with CAR for different windows
        """
        car_windows = {}
        
        # Define CAR windows
        windows = {
            'car_0_1': (0, 1),    # Event day + 1
            'car_0_5': (0, 5),    # Event day + 5
            'car_m1_1': (-1, 1),  # Day before to day after
            'car_m5_5': (-5, 5)   # 5 days before to 5 days after
        }
        
        for window_name, (start_day, end_day) in windows.items():
            # Calculate relative days from event
            event_date = event_ar['event_date'].iloc[0]
            event_ar_copy = event_ar.copy()
            event_ar_copy['relative_day'] = (event_ar_copy['date'] - event_date).dt.days
            
            # Filter by window
            window_data = event_ar_copy[
                (event_ar_copy['relative_day'] >= start_day) & 
                (event_ar_copy['relative_day'] <= end_day)
            ]
            
            if len(window_data) > 0:
                car_windows[window_name] = window_data['abnormal_return'].sum()
            else:
                car_windows[window_name] = 0.0
        
        return car_windows
    
    def _calculate_bhar(
        self, 
        returns: pd.DataFrame,
        market_returns: pd.Series,
        event_calendar: pd.DataFrame,
        risk_free_rate: Optional[pd.Series] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate buy-and-hold abnormal returns (BHAR).
        
        Args:
            returns: DataFrame with stablecoin returns
            market_returns: Series with market returns
            event_calendar: DataFrame with event calendar
            risk_free_rate: Optional Series with risk-free rate
            
        Returns:
            Dictionary with BHAR results
        """
        logger.info("Calculating buy-and-hold abnormal returns")
        
        bhar_results = {}
        
        for stablecoin in returns.columns:
            logger.info(f"Calculating BHAR for {stablecoin}")
            
            bhar_data = []
            
            for _, event in event_calendar.iterrows():
                event_date = event['event_date']
                event_start = event['event_window_start']
                event_end = event['event_window_end']
                
                # Get event window data
                event_returns = returns[stablecoin].loc[event_start:event_end]
                event_market = market_returns.loc[event_start:event_end]
                
                if len(event_returns) < 2:
                    continue
                
                # Calculate BHAR for different holding periods
                holding_periods = [1, 5, 10, 20, 60]  # Trading days
                
                bhar_windows = {}
                for period in holding_periods:
                    if len(event_returns) >= period:
                        # Asset BHAR
                        asset_bhar = (1 + event_returns.iloc[:period]).prod() - 1
                        
                        # Market BHAR
                        market_bhar = (1 + event_market.iloc[:period]).prod() - 1
                        
                        # Abnormal BHAR
                        abnormal_bhar = asset_bhar - market_bhar
                        
                        bhar_windows[f'bhar_{period}d'] = abnormal_bhar
                    else:
                        bhar_windows[f'bhar_{period}d'] = np.nan
                
                bhar_data.append({
                    'event_id': event.name,
                    'event_date': event_date,
                    **bhar_windows
                })
            
            bhar_results[stablecoin] = pd.DataFrame(bhar_data)
        
        return bhar_results
    
    def _run_statistical_tests(
        self, 
        car_results: Dict[str, pd.DataFrame],
        bhar_results: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        Run statistical tests on CAR and BHAR results.
        
        Args:
            car_results: Dictionary with CAR results
            bhar_results: Dictionary with BHAR results
            
        Returns:
            Dictionary with statistical test results
        """
        logger.info("Running statistical tests")
        
        test_results = {}
        
        # Test CAR results
        for stablecoin, car_data in car_results.items():
            if car_data.empty:
                continue
            
            car_tests = {}
            
            for col in car_data.columns:
                if col.startswith('car_'):
                    car_values = car_data[col].dropna()
                    
                    if len(car_values) > 0:
                        # t-test
                        t_stat, t_pvalue = stats.ttest_1samp(car_values, 0)
                        
                        # Wilcoxon signed-rank test
                        try:
                            w_stat, w_pvalue = stats.wilcoxon(car_values)
                        except ValueError:
                            w_stat, w_pvalue = np.nan, np.nan
                        
                        car_tests[col] = {
                            'mean': car_values.mean(),
                            'std': car_values.std(),
                            't_statistic': t_stat,
                            't_pvalue': t_pvalue,
                            'wilcoxon_statistic': w_stat,
                            'wilcoxon_pvalue': w_pvalue,
                            'n_observations': len(car_values)
                        }
            
            test_results[f'{stablecoin}_car'] = car_tests
        
        # Test BHAR results
        for stablecoin, bhar_data in bhar_results.items():
            if bhar_data.empty:
                continue
            
            bhar_tests = {}
            
            for col in bhar_data.columns:
                if col.startswith('bhar_'):
                    bhar_values = bhar_data[col].dropna()
                    
                    if len(bhar_values) > 0:
                        # t-test
                        t_stat, t_pvalue = stats.ttest_1samp(bhar_values, 0)
                        
                        # Wilcoxon signed-rank test
                        try:
                            w_stat, w_pvalue = stats.wilcoxon(bhar_values)
                        except ValueError:
                            w_stat, w_pvalue = np.nan, np.nan
                        
                        bhar_tests[col] = {
                            'mean': bhar_values.mean(),
                            'std': bhar_values.std(),
                            't_statistic': t_stat,
                            't_pvalue': t_pvalue,
                            'wilcoxon_statistic': w_stat,
                            'wilcoxon_pvalue': w_pvalue,
                            'n_observations': len(bhar_values)
                        }
            
            test_results[f'{stablecoin}_bhar'] = bhar_tests
        
        return test_results
    
    def _calculate_summary_statistics(
        self, 
        car_results: Dict[str, pd.DataFrame],
        bhar_results: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        Calculate summary statistics for event study results.
        
        Args:
            car_results: Dictionary with CAR results
            bhar_results: Dictionary with BHAR results
            
        Returns:
            Dictionary with summary statistics
        """
        logger.info("Calculating summary statistics")
        
        summary = {}
        
        # CAR summary
        car_summary = {}
        for stablecoin, car_data in car_results.items():
            if car_data.empty:
                continue
            
            stablecoin_summary = {}
            for col in car_data.columns:
                if col.startswith('car_'):
                    values = car_data[col].dropna()
                    if len(values) > 0:
                        stablecoin_summary[col] = {
                            'mean': values.mean(),
                            'median': values.median(),
                            'std': values.std(),
                            'min': values.min(),
                            'max': values.max(),
                            'positive_rate': (values > 0).mean(),
                            'n_events': len(values)
                        }
            
            car_summary[stablecoin] = stablecoin_summary
        
        summary['car'] = car_summary
        
        # BHAR summary
        bhar_summary = {}
        for stablecoin, bhar_data in bhar_results.items():
            if bhar_data.empty:
                continue
            
            stablecoin_summary = {}
            for col in bhar_data.columns:
                if col.startswith('bhar_'):
                    values = bhar_data[col].dropna()
                    if len(values) > 0:
                        stablecoin_summary[col] = {
                            'mean': values.mean(),
                            'median': values.median(),
                            'std': values.std(),
                            'min': values.min(),
                            'max': values.max(),
                            'positive_rate': (values > 0).mean(),
                            'n_events': len(values)
                        }
            
            bhar_summary[stablecoin] = stablecoin_summary
        
        summary['bhar'] = bhar_summary
        
        return summary
    
    def analyze_by_sentiment(
        self, 
        event_study_results: Dict,
        sentiment_data: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Analyze event study results by sentiment categories.
        
        Args:
            event_study_results: Results from event study analysis
            sentiment_data: DataFrame with sentiment analysis
            
        Returns:
            Dictionary with sentiment-based analysis
        """
        logger.info("Analyzing event study results by sentiment")
        
        sentiment_analysis = {}
        
        # Get sentiment categories
        sentiment_categories = sentiment_data['consensus_sentiment'].unique() if 'consensus_sentiment' in sentiment_data.columns else ['neutral']
        
        for sentiment in sentiment_categories:
            if sentiment not in ['hawkish', 'dovish', 'neutral']:
                continue
            
            # Filter events by sentiment
            sentiment_events = sentiment_data[sentiment_data['consensus_sentiment'] == sentiment]
            
            if len(sentiment_events) == 0:
                continue
            
            # Analyze CAR by sentiment
            sentiment_car = {}
            for stablecoin, car_data in event_study_results['car'].items():
                if car_data.empty:
                    continue
                
                # Filter CAR data by sentiment events
                sentiment_event_ids = sentiment_events.index
                sentiment_car_data = car_data[car_data['event_id'].isin(sentiment_event_ids)]
                
                if len(sentiment_car_data) > 0:
                    sentiment_car[stablecoin] = sentiment_car_data
            
            sentiment_analysis[sentiment] = {
                'n_events': len(sentiment_events),
                'car_results': sentiment_car
            }
        
        return sentiment_analysis
