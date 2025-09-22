"""
Analysis pipeline for stablecoin policy research.
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
from models.event_study import EventStudyAnalyzer
from models.garch import GARCHModel
from models.var_irf import VARModel

logger = logging.getLogger(__name__)


def run_analysis_pipeline(config_path: str = "configs/config.yaml") -> Dict[str, Dict]:
    """
    Run complete analysis pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Starting analysis pipeline")
    
    # Load configuration
    config = load_config(config_path)
    
    # Ensure output directories exist
    ensure_directory(config['output']['results_dir'])
    ensure_directory(config['output']['figures_dir'])
    
    # Load engineered features
    logger.info("Loading engineered features")
    features = load_engineered_features(config)
    
    if not features:
        logger.error("No engineered features found. Run feature engineering first.")
        return {}
    
    # Initialize analysis components
    event_study_analyzer = EventStudyAnalyzer(config)
    garch_model = GARCHModel(config)
    var_model = VARModel(config)
    
    # Run analyses
    analysis_results = {}
    
    try:
        # 1. Event Study Analysis
        logger.info("Running event study analysis")
        event_study_results = run_event_study_analysis(event_study_analyzer, features)
        analysis_results['event_study'] = event_study_results
        
        # 2. GARCH Analysis
        logger.info("Running GARCH analysis")
        garch_results = run_garch_analysis(garch_model, features)
        analysis_results['garch'] = garch_results
        
        # 3. VAR Analysis
        logger.info("Running VAR analysis")
        var_results = run_var_analysis(var_model, features)
        analysis_results['var'] = var_results
        
        # 4. Combined Analysis
        logger.info("Running combined analysis")
        combined_results = run_combined_analysis(analysis_results, features)
        analysis_results['combined'] = combined_results
        
        # Save analysis results
        logger.info("Saving analysis results")
        save_analysis_results(analysis_results, config)
        
        logger.info("Analysis pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {e}")
        raise
    
    return analysis_results


def load_engineered_features(config: Dict) -> Dict[str, pd.DataFrame]:
    """
    Load engineered features from files.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with loaded features
    """
    processed_dir = config['data']['processed_dir']
    features = {}
    
    # List of expected feature files
    expected_files = [
        'event_calendar',
        'returns',
        'volatility',
        'peg_deviations',
        'peg_deviation_probability',
        'rolling_correlation',
        'volatility_regime',
        'market_depth',
        'amihud',
        'policy_events_with_sentiment',
        'sentiment_timeseries',
        'market_sentiment_combined',
        'event_dummies',
        'hawkish_event_dummies',
        'dovish_event_dummies',
        'neutral_event_dummies'
    ]
    
    for filename in expected_files:
        try:
            data = load_data(filename, processed_dir)
            if not data.empty:
                features[filename] = data
                logger.info(f"Loaded {filename} with shape {data.shape}")
        except FileNotFoundError:
            logger.warning(f"File {filename} not found")
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
    
    return features


def run_event_study_analysis(
    analyzer: EventStudyAnalyzer, 
    features: Dict[str, pd.DataFrame]
) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Run event study analysis.
    
    Args:
        analyzer: Event study analyzer instance
        features: Dictionary with engineered features
        
    Returns:
        Dictionary with event study results
    """
    event_study_results = {}
    
    # Get required data
    returns = features.get('returns', pd.DataFrame())
    event_calendar = features.get('event_calendar', pd.DataFrame())
    
    if returns.empty or event_calendar.empty:
        logger.warning("Insufficient data for event study analysis")
        return event_study_results
    
    # Create market returns (use S&P 500 proxy or first stablecoin as benchmark)
    market_returns = returns.iloc[:, 0] if len(returns.columns) > 0 else pd.Series()
    
    if market_returns.empty:
        logger.warning("No market returns available")
        return event_study_results
    
    try:
        # Run event study
        results = analyzer.run_event_study(
            returns=returns,
            market_returns=market_returns,
            event_calendar=event_calendar
        )
        
        event_study_results.update(results)
        
        # Analyze by sentiment if available
        sentiment_events = features.get('policy_events_with_sentiment', pd.DataFrame())
        if not sentiment_events.empty:
            sentiment_analysis = analyzer.analyze_by_sentiment(results, sentiment_events)
            event_study_results['sentiment_analysis'] = sentiment_analysis
        
        logger.info("Event study analysis completed")
        
    except Exception as e:
        logger.error(f"Error in event study analysis: {e}")
    
    return event_study_results


def run_garch_analysis(
    model: GARCHModel, 
    features: Dict[str, pd.DataFrame]
) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Run GARCH analysis.
    
    Args:
        model: GARCH model instance
        features: Dictionary with engineered features
        
    Returns:
        Dictionary with GARCH results
    """
    garch_results = {}
    
    # Get required data
    returns = features.get('returns', pd.DataFrame())
    event_calendar = features.get('event_calendar', pd.DataFrame())
    
    if returns.empty:
        logger.warning("No returns data available for GARCH analysis")
        return garch_results
    
    try:
        # Fit GARCH models
        fitted_models = model.fit_garch_models(returns)
        garch_results['fitted_models'] = fitted_models
        
        # Calculate abnormal volatility
        if not event_calendar.empty:
            abnormal_volatility = model.calculate_abnormal_volatility(fitted_models, event_calendar)
            garch_results['abnormal_volatility'] = abnormal_volatility
        
        # Analyze volatility regimes
        volatility_regimes = model.analyze_volatility_regimes(returns)
        garch_results['volatility_regimes'] = volatility_regimes
        
        # Get model summary
        model_summary = model.get_model_summary(fitted_models)
        garch_results['model_summary'] = model_summary
        
        logger.info("GARCH analysis completed")
        
    except Exception as e:
        logger.error(f"Error in GARCH analysis: {e}")
    
    return garch_results


def run_var_analysis(
    model: VARModel, 
    features: Dict[str, pd.DataFrame]
) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Run VAR analysis.
    
    Args:
        model: VAR model instance
        features: Dictionary with engineered features
        
    Returns:
        Dictionary with VAR results
    """
    var_results = {}
    
    # Prepare VAR data
    var_data = prepare_var_data(features)
    
    if var_data.empty:
        logger.warning("No suitable data available for VAR analysis")
        return var_results
    
    try:
        # Fit VAR model
        fitted_models = model.fit_var_model(var_data)
        var_results['fitted_models'] = fitted_models
        
        if fitted_models:
            # Calculate impulse response functions
            irf_results = model.calculate_impulse_responses(fitted_models)
            var_results['impulse_responses'] = irf_results
            
            # Analyze policy transmission
            policy_transmission = model.analyze_policy_transmission(var_data)
            var_results['policy_transmission'] = policy_transmission
            
            # Test Granger causality
            causality_tests = model.test_granger_causality(fitted_models, 'fed_funds', ['stablecoin_returns'])
            var_results['granger_causality'] = causality_tests
            
            # Get model summary
            model_summary = model.get_model_summary(fitted_models)
            var_results['model_summary'] = model_summary
        
        logger.info("VAR analysis completed")
        
    except Exception as e:
        logger.error(f"Error in VAR analysis: {e}")
    
    return var_results


def prepare_var_data(features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Prepare data for VAR analysis.
    
    Args:
        features: Dictionary with engineered features
        
    Returns:
        DataFrame prepared for VAR analysis
    """
    var_data = pd.DataFrame()
    
    try:
        # Combine relevant variables
        components = []
        
        # Add returns data
        returns = features.get('returns', pd.DataFrame())
        if not returns.empty:
            # Use average returns across stablecoins
            avg_returns = returns.mean(axis=1)
            avg_returns.name = 'stablecoin_returns'
            components.append(avg_returns)
        
        # Add sentiment data
        sentiment_ts = features.get('sentiment_timeseries', pd.DataFrame())
        if not sentiment_ts.empty and 'sentiment_score' in sentiment_ts.columns:
            components.append(sentiment_ts['sentiment_score'])
        
        # Add volatility data
        volatility = features.get('volatility', pd.DataFrame())
        if not volatility.empty:
            avg_volatility = volatility.mean(axis=1)
            avg_volatility.name = 'stablecoin_volatility'
            components.append(avg_volatility)
        
        # Combine components
        if components:
            var_data = pd.concat(components, axis=1)
            var_data = var_data.dropna()
            
            logger.info(f"Prepared VAR data with {len(var_data)} observations and {len(var_data.columns)} variables")
        
    except Exception as e:
        logger.error(f"Error preparing VAR data: {e}")
    
    return var_data


def run_combined_analysis(
    analysis_results: Dict[str, Dict], 
    features: Dict[str, pd.DataFrame]
) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Run combined analysis across different methods.
    
    Args:
        analysis_results: Dictionary with individual analysis results
        features: Dictionary with engineered features
        
    Returns:
        Dictionary with combined analysis results
    """
    combined_results = {}
    
    try:
        # Combine event study and GARCH results
        if 'event_study' in analysis_results and 'garch' in analysis_results:
            event_garch_combined = combine_event_garch_results(
                analysis_results['event_study'],
                analysis_results['garch']
            )
            combined_results['event_garch_combined'] = event_garch_combined
        
        # Create comprehensive summary
        comprehensive_summary = create_comprehensive_summary(analysis_results, features)
        combined_results['comprehensive_summary'] = comprehensive_summary
        
        # Generate key findings
        key_findings = generate_key_findings(analysis_results, features)
        combined_results['key_findings'] = key_findings
        
        logger.info("Combined analysis completed")
        
    except Exception as e:
        logger.error(f"Error in combined analysis: {e}")
    
    return combined_results


def combine_event_garch_results(
    event_study_results: Dict, 
    garch_results: Dict
) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Combine event study and GARCH results.
    
    Args:
        event_study_results: Event study analysis results
        garch_results: GARCH analysis results
        
    Returns:
        Dictionary with combined results
    """
    combined_results = {}
    
    try:
        # Combine CAR and abnormal volatility results
        car_results = event_study_results.get('car', {})
        abnormal_volatility = garch_results.get('abnormal_volatility', {})
        
        for stablecoin in car_results.keys():
            if stablecoin in abnormal_volatility:
                car_data = car_results[stablecoin]
                av_data = abnormal_volatility[stablecoin]
                
                # Merge on event_id
                if not car_data.empty and not av_data.empty:
                    merged_data = car_data.merge(
                        av_data[['event_id', 'abnormal_volatility', 'volatility_ratio']],
                        on='event_id',
                        how='inner'
                    )
                    
                    combined_results[f'{stablecoin}_car_volatility'] = merged_data
        
    except Exception as e:
        logger.error(f"Error combining event study and GARCH results: {e}")
    
    return combined_results


def create_comprehensive_summary(
    analysis_results: Dict[str, Dict], 
    features: Dict[str, pd.DataFrame]
) -> Dict[str, Union[float, int, str]]:
    """
    Create comprehensive summary of all analyses.
    
    Args:
        analysis_results: Dictionary with analysis results
        features: Dictionary with engineered features
        
    Returns:
        Dictionary with comprehensive summary
    """
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'total_events': len(features.get('event_calendar', pd.DataFrame())),
        'analysis_period': {
            'start': features.get('returns', pd.DataFrame()).index.min().isoformat() if not features.get('returns', pd.DataFrame()).empty else None,
            'end': features.get('returns', pd.DataFrame()).index.max().isoformat() if not features.get('returns', pd.DataFrame()).empty else None
        }
    }
    
    # Event study summary
    if 'event_study' in analysis_results:
        event_summary = analysis_results['event_study'].get('summary', {})
        summary['event_study_summary'] = event_summary
    
    # GARCH summary
    if 'garch' in analysis_results:
        garch_summary = analysis_results['garch'].get('model_summary', {})
        summary['garch_summary'] = garch_summary
    
    # VAR summary
    if 'var' in analysis_results:
        var_summary = analysis_results['var'].get('model_summary', {})
        summary['var_summary'] = var_summary
    
    return summary


def generate_key_findings(
    analysis_results: Dict[str, Dict], 
    features: Dict[str, pd.DataFrame]
) -> Dict[str, str]:
    """
    Generate key findings from analysis results.
    
    Args:
        analysis_results: Dictionary with analysis results
        features: Dictionary with engineered features
        
    Returns:
        Dictionary with key findings
    """
    findings = {}
    
    try:
        # Event study findings
        if 'event_study' in analysis_results:
            car_summary = analysis_results['event_study'].get('summary', {}).get('car', {})
            
            for stablecoin, car_data in car_summary.items():
                if car_data:
                    car_0_1 = car_data.get('car_0_1', {})
                    if car_0_1:
                        mean_car = car_0_1.get('mean', 0)
                        positive_rate = car_0_1.get('positive_rate', 0)
                        
                        findings[f'{stablecoin}_event_impact'] = (
                            f"{stablecoin} shows {mean_car:.4f} average CAR(0,1) "
                            f"with {positive_rate:.1%} positive event rate"
                        )
        
        # Sentiment findings
        sentiment_events = features.get('policy_events_with_sentiment', pd.DataFrame())
        if not sentiment_events.empty and 'consensus_sentiment' in sentiment_events.columns:
            sentiment_dist = sentiment_events['consensus_sentiment'].value_counts()
            findings['sentiment_distribution'] = f"Policy events: {dict(sentiment_dist)}"
        
    except Exception as e:
        logger.error(f"Error generating key findings: {e}")
    
    return findings


def save_analysis_results(results: Dict[str, Dict], config: Dict) -> None:
    """
    Save analysis results to files.
    
    Args:
        results: Dictionary with analysis results
        config: Configuration dictionary
    """
    results_dir = config['output']['results_dir']
    
    for analysis_type, analysis_data in results.items():
        if isinstance(analysis_data, dict):
            for result_name, result_data in analysis_data.items():
                if isinstance(result_data, pd.DataFrame) and not result_data.empty:
                    try:
                        filename = f"{analysis_type}_{result_name}"
                        save_data(result_data, filename, results_dir, file_format='parquet')
                        logger.info(f"Saved {filename} with shape {result_data.shape}")
                    except Exception as e:
                        logger.error(f"Error saving {filename}: {e}")
                elif isinstance(result_data, dict):
                    try:
                        filename = f"{analysis_type}_{result_name}"
                        save_data(result_data, filename, results_dir, file_format='json')
                        logger.info(f"Saved {filename} as JSON")
                    except Exception as e:
                        logger.error(f"Error saving {filename}: {e}")


def main():
    """Main function for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run analysis pipeline')
    parser.add_argument('--config', default='configs/config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Run pipeline
    try:
        analysis_results = run_analysis_pipeline(args.config)
        print(f"Successfully completed analysis with {len(analysis_results)} result sets")
    except Exception as e:
        print(f"Error in analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
