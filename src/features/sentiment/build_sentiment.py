"""
Orchestrates sentiment analysis using multiple methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from .llm_adapter import LLMAdapter
from .finbert import FinBERTAnalyzer

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Orchestrates sentiment analysis using multiple methods.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize sentiment analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize analyzers
        self.llm_adapter = None
        self.finbert_analyzer = None
        
        # Get API keys
        api_keys = config.get('api_keys', {})
        deepseek_key = api_keys.get('deepseek')
        
        if deepseek_key:
            self.llm_adapter = LLMAdapter(
                api_key=deepseek_key,
                model=config.get('sentiment', {}).get('model', 'deepseek-chat')
            )
        
        # Initialize FinBERT (no API key needed)
        try:
            self.finbert_analyzer = FinBERTAnalyzer()
        except Exception as e:
            logger.warning(f"Could not initialize FinBERT: {e}")
            self.finbert_analyzer = None
    
    def analyze_sentiment_comprehensive(
        self, 
        events_df: pd.DataFrame,
        text_column: str = 'content',
        methods: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Perform comprehensive sentiment analysis using multiple methods.
        
        Args:
            events_df: DataFrame with policy events
            text_column: Column containing text to analyze
            methods: List of methods to use ('llm', 'finbert', 'both')
            
        Returns:
            DataFrame with comprehensive sentiment analysis
        """
        if methods is None:
            methods = ['both']
        
        logger.info(f"Performing comprehensive sentiment analysis using methods: {methods}")
        
        result_df = events_df.copy()
        
        # LLM Analysis
        if 'llm' in methods or 'both' in methods:
            if self.llm_adapter:
                logger.info("Running LLM sentiment analysis")
                llm_results = self.llm_adapter.analyze_policy_events(events_df, text_column)
                
                # Merge LLM results
                llm_columns = ['sentiment', 'sentiment_confidence', 'sentiment_key_phrases', 'sentiment_explanation']
                for col in llm_columns:
                    if col in llm_results.columns:
                        result_df[f'{col}_llm'] = llm_results[col]
            else:
                logger.warning("LLM adapter not available")
        
        # FinBERT Analysis
        if 'finbert' in methods or 'both' in methods:
            if self.finbert_analyzer:
                logger.info("Running FinBERT sentiment analysis")
                finbert_results = self.finbert_analyzer.analyze_policy_events(events_df, text_column)
                
                # Merge FinBERT results
                finbert_columns = ['sentiment_finbert', 'sentiment_confidence_finbert', 'sentiment_probabilities_finbert']
                for col in finbert_columns:
                    if col in finbert_results.columns:
                        result_df[col] = finbert_results[col]
            else:
                logger.warning("FinBERT analyzer not available")
        
        # Create consensus sentiment
        result_df = self._create_consensus_sentiment(result_df)
        
        return result_df
    
    def _create_consensus_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create consensus sentiment from multiple methods.
        
        Args:
            df: DataFrame with sentiment results from multiple methods
            
        Returns:
            DataFrame with consensus sentiment
        """
        logger.info("Creating consensus sentiment")
        
        # Determine which sentiment columns are available
        sentiment_columns = []
        if 'sentiment_llm' in df.columns:
            sentiment_columns.append('sentiment_llm')
        if 'sentiment_finbert' in df.columns:
            sentiment_columns.append('sentiment_finbert')
        
        if len(sentiment_columns) == 0:
            logger.warning("No sentiment columns found for consensus")
            df['consensus_sentiment'] = 'neutral'
            df['consensus_confidence'] = 0.0
            return df
        
        # Create consensus sentiment
        consensus_sentiments = []
        consensus_confidences = []
        
        for idx, row in df.iterrows():
            sentiments = []
            confidences = []
            
            # Collect sentiments and confidences
            if 'sentiment_llm' in df.columns and pd.notna(row['sentiment_llm']):
                sentiments.append(row['sentiment_llm'])
                confidences.append(row.get('sentiment_confidence_llm', 0.5))
            
            if 'sentiment_finbert' in df.columns and pd.notna(row['sentiment_finbert']):
                sentiments.append(row['sentiment_finbert'])
                confidences.append(row.get('sentiment_confidence_finbert', 0.5))
            
            # Determine consensus
            if len(sentiments) == 0:
                consensus_sentiment = 'neutral'
                consensus_confidence = 0.0
            elif len(sentiments) == 1:
                consensus_sentiment = sentiments[0]
                consensus_confidence = confidences[0]
            else:
                # Multiple methods - use weighted average
                consensus_sentiment, consensus_confidence = self._weighted_consensus(sentiments, confidences)
            
            consensus_sentiments.append(consensus_sentiment)
            consensus_confidences.append(consensus_confidence)
        
        df['consensus_sentiment'] = consensus_sentiments
        df['consensus_confidence'] = consensus_confidences
        
        return df
    
    def _weighted_consensus(self, sentiments: List[str], confidences: List[float]) -> tuple:
        """
        Calculate weighted consensus from multiple sentiment predictions.
        
        Args:
            sentiments: List of sentiment predictions
            confidences: List of confidence scores
            
        Returns:
            Tuple of (consensus_sentiment, consensus_confidence)
        """
        # Count sentiment votes weighted by confidence
        sentiment_weights = {'hawkish': 0.0, 'dovish': 0.0, 'neutral': 0.0}
        
        for sentiment, confidence in zip(sentiments, confidences):
            if sentiment in sentiment_weights:
                sentiment_weights[sentiment] += confidence
        
        # Find sentiment with highest weighted score
        consensus_sentiment = max(sentiment_weights, key=sentiment_weights.get)
        
        # Calculate consensus confidence
        total_weight = sum(sentiment_weights.values())
        consensus_confidence = sentiment_weights[consensus_sentiment] / total_weight if total_weight > 0 else 0.0
        
        return consensus_sentiment, consensus_confidence
    
    def analyze_sentiment_timeseries(
        self, 
        events_df: pd.DataFrame,
        freq: str = 'D'
    ) -> pd.DataFrame:
        """
        Create sentiment time series from events.
        
        Args:
            events_df: DataFrame with sentiment-analyzed events
            freq: Frequency for time series ('D', 'W', 'M')
            
        Returns:
            DataFrame with sentiment time series
        """
        logger.info(f"Creating sentiment time series with frequency {freq}")
        
        if events_df.empty:
            return pd.DataFrame()
        
        # Ensure date column is datetime
        events_df = events_df.copy()
        events_df['date'] = pd.to_datetime(events_df['date'])
        
        # Create sentiment time series
        sentiment_ts = []
        
        # Get date range
        start_date = events_df['date'].min()
        end_date = events_df['date'].max()
        date_range = pd.date_range(start_date, end_date, freq=freq)
        
        for date in date_range:
            # Find events within the period
            period_events = events_df[
                (events_df['date'] >= date) & 
                (events_df['date'] < date + pd.Timedelta(days=1 if freq == 'D' else 7 if freq == 'W' else 30))
            ]
            
            if len(period_events) > 0:
                # Calculate period sentiment metrics
                period_sentiment = self._calculate_period_sentiment(period_events)
                period_sentiment['date'] = date
                sentiment_ts.append(period_sentiment)
            else:
                # No events in period
                sentiment_ts.append({
                    'date': date,
                    'sentiment_score': 0.0,
                    'sentiment_volatility': 0.0,
                    'event_count': 0,
                    'hawkish_count': 0,
                    'dovish_count': 0,
                    'neutral_count': 0
                })
        
        sentiment_df = pd.DataFrame(sentiment_ts)
        sentiment_df = sentiment_df.set_index('date')
        
        return sentiment_df
    
    def _calculate_period_sentiment(self, period_events: pd.DataFrame) -> Dict:
        """
        Calculate sentiment metrics for a time period.
        
        Args:
            period_events: Events within the time period
            
        Returns:
            Dictionary with period sentiment metrics
        """
        # Use consensus sentiment if available, otherwise fall back to individual methods
        sentiment_col = 'consensus_sentiment' if 'consensus_sentiment' in period_events.columns else 'sentiment_llm'
        confidence_col = 'consensus_confidence' if 'consensus_confidence' in period_events.columns else 'sentiment_confidence_llm'
        
        if sentiment_col not in period_events.columns:
            sentiment_col = 'sentiment_finbert'
            confidence_col = 'sentiment_confidence_finbert'
        
        if sentiment_col not in period_events.columns:
            # No sentiment data available
            return {
                'sentiment_score': 0.0,
                'sentiment_volatility': 0.0,
                'event_count': len(period_events),
                'hawkish_count': 0,
                'dovish_count': 0,
                'neutral_count': 0
            }
        
        # Calculate sentiment score (-1 to 1: dovish to hawkish)
        sentiment_scores = []
        for _, event in period_events.iterrows():
            sentiment = event[sentiment_col]
            confidence = event.get(confidence_col, 0.5)
            
            if sentiment == 'hawkish':
                score = confidence
            elif sentiment == 'dovish':
                score = -confidence
            else:  # neutral
                score = 0.0
            
            sentiment_scores.append(score)
        
        # Calculate metrics
        avg_sentiment_score = np.mean(sentiment_scores) if sentiment_scores else 0.0
        sentiment_volatility = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.0
        
        # Count sentiments
        sentiment_counts = period_events[sentiment_col].value_counts()
        hawkish_count = sentiment_counts.get('hawkish', 0)
        dovish_count = sentiment_counts.get('dovish', 0)
        neutral_count = sentiment_counts.get('neutral', 0)
        
        return {
            'sentiment_score': avg_sentiment_score,
            'sentiment_volatility': sentiment_volatility,
            'event_count': len(period_events),
            'hawkish_count': hawkish_count,
            'dovish_count': dovish_count,
            'neutral_count': neutral_count
        }
    
    def get_sentiment_summary(self, sentiment_df: pd.DataFrame) -> Dict:
        """
        Get comprehensive summary of sentiment analysis.
        
        Args:
            sentiment_df: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with comprehensive summary
        """
        logger.info("Generating sentiment analysis summary")
        
        summary = {
            'total_events': len(sentiment_df),
            'analysis_methods': []
        }
        
        # Check which methods were used
        if 'sentiment_llm' in sentiment_df.columns:
            summary['analysis_methods'].append('LLM')
            summary['llm_summary'] = self._get_method_summary(sentiment_df, 'llm')
        
        if 'sentiment_finbert' in sentiment_df.columns:
            summary['analysis_methods'].append('FinBERT')
            summary['finbert_summary'] = self._get_method_summary(sentiment_df, 'finbert')
        
        if 'consensus_sentiment' in sentiment_df.columns:
            summary['analysis_methods'].append('Consensus')
            summary['consensus_summary'] = self._get_method_summary(sentiment_df, 'consensus')
        
        return summary
    
    def _get_method_summary(self, df: pd.DataFrame, method: str) -> Dict:
        """
        Get summary for a specific sentiment analysis method.
        
        Args:
            df: DataFrame with sentiment results
            method: Method name ('llm', 'finbert', 'consensus')
            
        Returns:
            Dictionary with method summary
        """
        sentiment_col = f'sentiment_{method}' if method != 'consensus' else 'consensus_sentiment'
        confidence_col = f'sentiment_confidence_{method}' if method != 'consensus' else 'consensus_confidence'
        
        if sentiment_col not in df.columns:
            return {}
        
        summary = {
            'sentiment_distribution': df[sentiment_col].value_counts().to_dict(),
            'average_confidence': df[confidence_col].mean() if confidence_col in df.columns else 0.0,
            'high_confidence_rate': (df[confidence_col] >= 0.7).mean() if confidence_col in df.columns else 0.0
        }
        
        return summary
