"""
VAR models with impulse response functions for policy transmission analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.irf import IRAnalysis
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class VARModel:
    """
    VAR models with impulse response functions for policy transmission.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize VAR model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.lags = config.get('analysis', {}).get('var', {}).get('lags', 4)
        self.variables = config.get('analysis', {}).get('var', {}).get('variables', ['fed_funds', 'stablecoin_supply', 'peg_deviation'])
    
    def fit_var_model(
        self, 
        data: pd.DataFrame,
        variables: Optional[List[str]] = None
    ) -> Dict[str, VAR]:
        """
        Fit VAR models to data.
        
        Args:
            data: DataFrame with time series data
            variables: List of variables to include in VAR
            
        Returns:
            Dictionary with fitted VAR models
        """
        logger.info("Fitting VAR models")
        
        if variables is None:
            variables = self.variables
        
        fitted_models = {}
        
        # Check which variables are available in data
        available_vars = [var for var in variables if var in data.columns]
        
        if len(available_vars) < 2:
            logger.warning("Insufficient variables for VAR model")
            return fitted_models
        
        logger.info(f"Fitting VAR with variables: {available_vars}")
        
        try:
            # Prepare data
            var_data = data[available_vars].dropna()
            
            if len(var_data) < 50:  # Need minimum observations
                logger.warning("Insufficient observations for VAR model")
                return fitted_models
            
            # Fit VAR model
            model = VAR(var_data)
            fitted_model = model.fit(maxlags=self.lags, ic='aic')
            
            fitted_models['main'] = fitted_model
            
            logger.info(f"VAR model fitted with {fitted_model.k_ar} lags")
            
        except Exception as e:
            logger.error(f"Error fitting VAR model: {e}")
        
        return fitted_models
    
    def calculate_impulse_responses(
        self, 
        fitted_models: Dict[str, VAR],
        periods: int = 20
    ) -> Dict[str, Dict]:
        """
        Calculate impulse response functions.
        
        Args:
            fitted_models: Dictionary with fitted VAR models
            periods: Number of periods for IRF
            
        Returns:
            Dictionary with impulse response results
        """
        logger.info("Calculating impulse response functions")
        
        irf_results = {}
        
        for model_name, model in fitted_models.items():
            try:
                logger.info(f"Calculating IRF for {model_name}")
                
                # Create IRF analysis
                irf = IRAnalysis(model)
                
                # Calculate impulse responses
                irf_data = irf.irf(periods)
                
                # Store results
                irf_results[model_name] = {
                    'irf': irf_data,
                    'orthogonalized': irf.orth_irf(periods),
                    'cumulative': irf.cum_effects(periods)
                }
                
            except Exception as e:
                logger.error(f"Error calculating IRF for {model_name}: {e}")
                continue
        
        return irf_results
    
    def analyze_policy_transmission(
        self, 
        data: pd.DataFrame,
        policy_variable: str = 'fed_funds',
        target_variables: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Analyze policy transmission effects.
        
        Args:
            data: DataFrame with time series data
            policy_variable: Policy variable (e.g., fed_funds)
            target_variables: Target variables to analyze
            
        Returns:
            Dictionary with policy transmission analysis
        """
        logger.info(f"Analyzing policy transmission from {policy_variable}")
        
        if target_variables is None:
            target_variables = [col for col in data.columns if col != policy_variable]
        
        # Ensure policy variable is in data
        if policy_variable not in data.columns:
            logger.error(f"Policy variable {policy_variable} not found in data")
            return {}
        
        # Prepare variables for VAR
        var_variables = [policy_variable] + target_variables
        var_data = data[var_variables].dropna()
        
        if len(var_data) < 50:
            logger.warning("Insufficient data for policy transmission analysis")
            return {}
        
        # Fit VAR model
        fitted_models = self.fit_var_model(var_data, var_variables)
        
        if not fitted_models:
            return {}
        
        # Calculate impulse responses
        irf_results = self.calculate_impulse_responses(fitted_models)
        
        # Analyze policy shocks
        transmission_analysis = self._analyze_policy_shocks(
            irf_results, policy_variable, target_variables
        )
        
        return transmission_analysis
    
    def _analyze_policy_shocks(
        self, 
        irf_results: Dict,
        policy_variable: str,
        target_variables: List[str]
    ) -> Dict[str, Dict]:
        """
        Analyze the effects of policy shocks.
        
        Args:
            irf_results: IRF results
            policy_variable: Policy variable
            target_variables: Target variables
            
        Returns:
            Dictionary with policy shock analysis
        """
        logger.info("Analyzing policy shock effects")
        
        shock_analysis = {}
        
        for model_name, irf_data in irf_results.items():
            try:
                irf = irf_data['irf']
                
                # Get policy shock effects
                policy_shock_effects = {}
                
                for target_var in target_variables:
                    if target_var in irf.columns:
                        # Get impulse response to policy shock
                        shock_response = irf[target_var].loc[policy_variable]
                        
                        # Calculate summary statistics
                        max_response = shock_response.max()
                        min_response = shock_response.min()
                        cumulative_response = shock_response.sum()
                        
                        # Find peak response
                        peak_period = shock_response.idxmax()
                        
                        policy_shock_effects[target_var] = {
                            'max_response': max_response,
                            'min_response': min_response,
                            'cumulative_response': cumulative_response,
                            'peak_period': peak_period,
                            'peak_response': max_response,
                            'response_series': shock_response
                        }
                
                shock_analysis[model_name] = policy_shock_effects
                
            except Exception as e:
                logger.error(f"Error analyzing policy shocks for {model_name}: {e}")
                continue
        
        return shock_analysis
    
    def forecast_policy_effects(
        self, 
        fitted_models: Dict[str, VAR],
        policy_shock: float,
        periods: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Forecast effects of policy shocks.
        
        Args:
            fitted_models: Dictionary with fitted VAR models
            policy_shock: Magnitude of policy shock
            periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        logger.info(f"Forecasting policy effects for shock: {policy_shock}")
        
        forecast_results = {}
        
        for model_name, model in fitted_models.items():
            try:
                # Get impulse response
                irf_results = self.calculate_impulse_responses({model_name: model}, periods)
                
                if model_name in irf_results:
                    irf = irf_results[model_name]['irf']
                    
                    # Scale impulse responses by shock magnitude
                    scaled_irf = irf * policy_shock
                    
                    forecast_results[model_name] = scaled_irf
                
            except Exception as e:
                logger.error(f"Error forecasting policy effects for {model_name}: {e}")
                continue
        
        return forecast_results
    
    def get_model_summary(self, fitted_models: Dict[str, VAR]) -> Dict[str, Dict]:
        """
        Get summary of fitted VAR models.
        
        Args:
            fitted_models: Dictionary with fitted models
            
        Returns:
            Dictionary with model summaries
        """
        logger.info("Generating VAR model summary")
        
        summary = {}
        
        for model_name, model in fitted_models.items():
            try:
                # Get model summary
                model_summary = model.summary()
                
                # Get information criteria
                ic_results = model.info_criteria
                
                # Get stability
                stability = model.is_stable()
                
                summary[model_name] = {
                    'summary': model_summary,
                    'information_criteria': ic_results,
                    'stability': stability,
                    'k_ar': model.k_ar,
                    'n_obs': model.nobs,
                    'variables': model.names
                }
                
            except Exception as e:
                logger.error(f"Error getting summary for {model_name}: {e}")
                continue
        
        return summary
    
    def test_granger_causality(
        self, 
        fitted_models: Dict[str, VAR],
        policy_variable: str,
        target_variables: List[str]
    ) -> Dict[str, Dict]:
        """
        Test Granger causality between variables.
        
        Args:
            fitted_models: Dictionary with fitted VAR models
            policy_variable: Policy variable
            target_variables: Target variables
            
        Returns:
            Dictionary with Granger causality test results
        """
        logger.info("Testing Granger causality")
        
        causality_results = {}
        
        for model_name, model in fitted_models.items():
            try:
                causality_tests = {}
                
                for target_var in target_variables:
                    if target_var in model.names:
                        # Test if policy variable Granger causes target variable
                        test_result = model.test_causality(
                            target_var, policy_variable, kind='f'
                        )
                        
                        causality_tests[f'{policy_variable}_to_{target_var}'] = {
                            'test_statistic': test_result.test_statistic,
                            'p_value': test_result.pvalue,
                            'critical_value': test_result.critical_value,
                            'conclusion': 'Reject H0' if test_result.pvalue < 0.05 else 'Fail to reject H0'
                        }
                
                causality_results[model_name] = causality_tests
                
            except Exception as e:
                logger.error(f"Error testing Granger causality for {model_name}: {e}")
                continue
        
        return causality_results
