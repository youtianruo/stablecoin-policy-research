"""
LLM adapter for sentiment analysis of policy text using DeepSeek API.
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
import time
import json

logger = logging.getLogger(__name__)


class LLMAdapter:
    """
    Unified interface for LLM-based sentiment analysis using DeepSeek API.
    """
    
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        """
        Initialize LLM adapter.
        
        Args:
            api_key: DeepSeek API key
            model: Model to use for analysis (default: deepseek-chat)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Sentiment categories
        self.sentiment_categories = [
            "hawkish",  # Tightening monetary policy
            "dovish",   # Accommodative monetary policy
            "neutral"   # Balanced or data-dependent
        ]
    
    def analyze_sentiment(
        self, 
        text: str, 
        context: str = "Federal Reserve monetary policy"
    ) -> Dict[str, Union[float, str]]:
        """
        Analyze sentiment of policy text using DeepSeek API.
        
        Args:
            text: Text to analyze
            context: Context for the analysis
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            prompt = self._create_sentiment_prompt(text, context)
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert financial analyst specializing in Federal Reserve policy analysis."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,  # Low temperature for consistent results
                "max_tokens": 500
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result_data = response.json()
            result_text = result_data['choices'][0]['message']['content']
            sentiment_result = self._parse_sentiment_response(result_text)
            
            return sentiment_result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._get_default_sentiment()
    
    def _create_sentiment_prompt(self, text: str, context: str) -> str:
        """
        Create prompt for sentiment analysis.
        """
        prompt = f"""
Analyze the sentiment of the following {context} text and provide a structured response.

Text to analyze:
{text[:4000]}  # Limit text length

Please analyze the sentiment and provide:
1. Overall sentiment category: hawkish, dovish, or neutral
2. Confidence score (0-1) for the sentiment classification
3. Key phrases that support your classification
4. Brief explanation of your reasoning

Respond in the following JSON format:
{{
    "sentiment": "hawkish/dovish/neutral",
    "confidence": 0.85,
    "key_phrases": ["phrase1", "phrase2"],
    "explanation": "Brief explanation"
}}
"""
        return prompt
    
    def _parse_sentiment_response(self, response_text: str) -> Dict[str, Union[float, str]]:
        """
        Parse LLM response into structured format.
        """
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Validate and clean result
                sentiment = result.get('sentiment', 'neutral').lower()
                if sentiment not in self.sentiment_categories:
                    sentiment = 'neutral'
                
                confidence = float(result.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                
                return {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'key_phrases': result.get('key_phrases', []),
                    'explanation': result.get('explanation', ''),
                    'raw_response': response_text
                }
            else:
                # Fallback parsing
                return self._parse_fallback_response(response_text)
                
        except Exception as e:
            logger.error(f"Error parsing sentiment response: {e}")
            return self._get_default_sentiment()
    
    def _parse_fallback_response(self, response_text: str) -> Dict[str, Union[float, str]]:
        """
        Fallback parsing for non-JSON responses.
        """
        sentiment = 'neutral'
        confidence = 0.5
        
        # Simple keyword matching
        text_lower = response_text.lower()
        
        if 'hawkish' in text_lower:
            sentiment = 'hawkish'
            confidence = 0.7
        elif 'dovish' in text_lower:
            sentiment = 'dovish'
            confidence = 0.7
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'key_phrases': [],
            'explanation': 'Fallback parsing used',
            'raw_response': response_text
        }
    
    def _get_default_sentiment(self) -> Dict[str, Union[float, str]]:
        """
        Get default sentiment result for error cases.
        """
        return {
            'sentiment': 'neutral',
            'confidence': 0.0,
            'key_phrases': [],
            'explanation': 'Error in analysis',
            'raw_response': ''
        }
    
    def analyze_batch(
        self, 
        texts: List[str], 
        contexts: Optional[List[str]] = None,
        delay: float = 0.1
    ) -> List[Dict[str, Union[float, str]]]:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
            contexts: Optional list of contexts
            delay: Delay between API calls
            
        Returns:
            List of sentiment analysis results
        """
        logger.info(f"Analyzing sentiment for {len(texts)} texts")
        
        results = []
        
        for i, text in enumerate(texts):
            context = contexts[i] if contexts and i < len(contexts) else "Federal Reserve monetary policy"
            
            try:
                result = self.analyze_sentiment(text, context)
                results.append(result)
                
                # Rate limiting
                if delay > 0:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Error analyzing text {i}: {e}")
                results.append(self._get_default_sentiment())
        
        return results
    
    def analyze_policy_events(
        self, 
        events_df: pd.DataFrame,
        text_column: str = 'content'
    ) -> pd.DataFrame:
        """
        Analyze sentiment for policy events DataFrame.
        
        Args:
            events_df: DataFrame with policy events
            text_column: Column containing text to analyze
            
        Returns:
            DataFrame with sentiment analysis results
        """
        logger.info(f"Analyzing sentiment for {len(events_df)} policy events")
        
        if events_df.empty:
            return pd.DataFrame()
        
        # Prepare texts for analysis
        texts = events_df[text_column].fillna('').tolist()
        
        # Analyze sentiment
        sentiment_results = self.analyze_batch(texts)
        
        # Add results to DataFrame
        result_df = events_df.copy()
        
        result_df['sentiment'] = [r['sentiment'] for r in sentiment_results]
        result_df['sentiment_confidence'] = [r['confidence'] for r in sentiment_results]
        result_df['sentiment_key_phrases'] = [r['key_phrases'] for r in sentiment_results]
        result_df['sentiment_explanation'] = [r['explanation'] for r in sentiment_results]
        
        return result_df
    
    def get_sentiment_summary(self, sentiment_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for sentiment analysis.
        
        Args:
            sentiment_df: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        if sentiment_df.empty:
            return {}
        
        summary = {
            'total_events': len(sentiment_df),
            'sentiment_distribution': sentiment_df['sentiment'].value_counts().to_dict(),
            'average_confidence': sentiment_df['sentiment_confidence'].mean(),
            'confidence_by_sentiment': sentiment_df.groupby('sentiment')['sentiment_confidence'].mean().to_dict()
        }
        
        return summary
    
    def validate_sentiment_results(self, sentiment_df: pd.DataFrame) -> Dict[str, int]:
        """
        Validate sentiment analysis results.
        
        Args:
            sentiment_df: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_events': len(sentiment_df),
            'valid_sentiments': len(sentiment_df[sentiment_df['sentiment'].isin(self.sentiment_categories)]),
            'high_confidence': len(sentiment_df[sentiment_df['sentiment_confidence'] >= 0.7]),
            'low_confidence': len(sentiment_df[sentiment_df['sentiment_confidence'] < 0.3])
        }
        
        validation_results['validation_rate'] = validation_results['valid_sentiments'] / validation_results['total_events'] if validation_results['total_events'] > 0 else 0
        
        return validation_results
