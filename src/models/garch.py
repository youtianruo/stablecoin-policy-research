"""
GARCH models for volatility analysis of stablecoin markets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from arch import arch_model
from arch.univariate import GARCH, EGARCH, GJR_GARCH
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class GARCHModel:
    """
    GARCH models for volatility analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize GARCH model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = config.get('analysis', {}).get('garch', {}).get('models', ['GARCH'])
        self.distribution = config.get('analysis', {}).get('garch', {}).get('distribution', 'normal')
    
    def fit_garch_models(self, returns: pd.DataFrame) -> Dict[str, Dict]:
        """
        Fit GARCH models to returns data.
        
        Args:
            returns: DataFrame with returns data
            
        Returns:
            Dictionary with fitted models
        """
        logger.info("Fitting GARCH models")
        
        fitted_models = {}
        
        for stablecoin in returns.columns:
            logger.info(f"Fitting GARCH models for {stablecoin}")
            
            stablecoin_returns = returns[stablecoin].dropna()
            
            if len(stablecoin_returns) < 100:  # Need minimum observations
                logger.warning(f"Insufficient data for {stablecoin}")
                continue
            
            stablecoin_models = {}
            
            for model_type in self.models:
                try:
                    model = self._fit_single_garch(stablecoin_returns, model_type)
                    stablecoin_models[model_type] = model
                except Exception as e:
                    logger.error(f"Error fitting {model_type} for {stablecoin}: {e}")
                    continue
            
            fitted_models[stablecoin] = stablecoin_models
        
        return fitted_models
    
    def _fit_single_garch(self, returns: pd.Series, model_type: str):
        """
        Fit a single GARCH model.
        
        Args:
            returns: Returns series
            model_type: Type of GARCH model
            
        Returns:
            Fitted GARCH model
        """
        if model_type == 'GARCH':
            model = arch_model(returns, vol='GARCH', p=1, q=1, dist=self.distribution)
        elif model_type == 'EGARCH':
            model = arch_model(returns, vol='EGARCH', p=1, q=1, dist=self.distribution)
        elif model_type == 'GJR-GARCH':
            model = arch_model(returns, vol='GARCH', p=1, o=1, q=1, dist=self.distribution)
        else:
            raise ValueError(f"Unknown GARCH model type: {model_type}")
        
        # Fit model
        fitted_model = model.fit(disp='off')
        
        return fitted_model
    
    def calculate_abnormal_volatility(
        self, 
        fitted_models: Dict[str, Dict],
        event_calendar: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate abnormal volatility around events.
        
        Args:
            fitted_models: Dictionary with fitted GARCH models
            event_calendar: DataFrame with event calendar
            
        Returns:
            Dictionary with abnormal volatility results
        """
        logger.info("Calculating abnormal volatility")
        
        abnormal_volatility = {}
        
        for stablecoin, models in fitted_models.items():
            if not models:
                continue
            
            # Use the first available model
            model_type = list(models.keys())[0]
            model = models[model_type]
            
            logger.info(f"Calculating abnormal volatility for {stablecoin} using {model_type}")
            
            av_data = []
            
            for _, event in event_calendar.iterrows():
                event_date = event['event_date']
                event_start = event['event_window_start']
                event_end = event['event_window_end']
                
                # Get event window dates
                event_dates = pd.date_range(event_start, event_end, freq='D')
                event_dates = event_dates[event_dates.isin(model.model.data.index)]
                
                if len(event_dates) == 0:
                    continue
                
                # Calculate expected volatility
                expected_vol = self._calculate_expected_volatility(model, event_dates)
                
                # Calculate actual volatility
                actual_vol = self._calculate_actual_volatility(model, event_dates)
                
                # Calculate abnormal volatility
                abnormal_vol = actual_vol - expected_vol
                
                av_data.append({
                    'event_id': event.name,
                    'event_date': event_date,
                    'abnormal_volatility': abnormal_vol.mean(),
                    'expected_volatility': expected_vol.mean(),
                    'actual_volatility': actual_vol.mean(),
                    'volatility_ratio': actual_vol.mean() / expected_vol.mean()
                })
            
            abnormal_volatility[stablecoin] = pd.DataFrame(av_data)
        
        return abnormal_volatility
    
    def _calculate_expected_volatility(self, model, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Calculate expected volatility from GARCH model.
        
        Args:
            model: Fitted GARCH model
            dates: Dates for volatility calculation
            
        Returns:
            Series with expected volatility
        """
        # Use the model's conditional volatility
        conditional_vol = model.conditional_volatility
        
        # Get volatility for the specified dates
        expected_vol = conditional_vol.loc[dates]
        
        return expected_vol
    
    def _calculate_actual_volatility(self, model, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Calculate actual volatility from residuals.
        
        Args:
            model: Fitted GARCH model
            dates: Dates for volatility calculation
            
        Returns:
            Series with actual volatility
        """
        # Get residuals
        residuals = model.resid
        
        # Calculate rolling volatility of residuals
        actual_vol = residuals.loc[dates].rolling(window=5).std()
        
        return actual_vol
    
    def analyze_volatility_regimes(self, returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Analyze volatility regimes using GARCH models.
        
        Args:
            returns: DataFrame with returns data
            
        Returns:
            Dictionary with volatility regime analysis
        """
        logger.info("Analyzing volatility regimes")
        
        regime_analysis = {}
        
        for stablecoin in returns.columns:
            stablecoin_returns = returns[stablecoin].dropna()
            
            if len(stablecoin_returns) < 100:
                continue
            
            try:
                # Fit GARCH model
                model = self._fit_single_garch(stablecoin_returns, 'GARCH')
                
                # Get conditional volatility
                conditional_vol = model.conditional_volatility
                
                # Identify volatility regimes
                vol_threshold = conditional_vol.quantile(0.75)
                high_vol_regime = conditional_vol > vol_threshold
                
                regime_data = pd.DataFrame({
                    'date': conditional_vol.index,
                    'conditional_volatility': conditional_vol,
                    'high_volatility_regime': high_vol_regime,
                    'returns': stablecoin_returns
                })
                
                regime_analysis[stablecoin] = regime_data
                
            except Exception as e:
                logger.error(f"Error analyzing volatility regimes for {stablecoin}: {e}")
                continue
        
        return regime_analysis
    
    def get_model_summary(self, fitted_models: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Get summary of fitted GARCH models.
        
        Args:
            fitted_models: Dictionary with fitted models
            
        Returns:
            Dictionary with model summaries
        """
        logger.info("Generating GARCH model summary")
        
        summary = {}
        
        for stablecoin, models in fitted_models.items():
            stablecoin_summary = {}
            
            for model_type, model in models.items():
                try:
                    # Get model parameters
                    params = model.params
                    
                    # Get model statistics
                    aic = model.aic
                    bic = model.bic
                    loglik = model.loglikelihood
                    
                    stablecoin_summary[model_type] = {
                        'parameters': params.to_dict(),
                        'aic': aic,
                        'bic': bic,
                        'loglikelihood': loglik,
                        'convergence': model.convergence_flag
                    }
                    
                except Exception as e:
                    logger.error(f"Error getting summary for {stablecoin} {model_type}: {e}")
                    continue
            
            summary[stablecoin] = stablecoin_summary
        
        return summary
