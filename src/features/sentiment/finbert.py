"""
FinBERT-based sentiment analysis for financial text.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FinBERTAnalyzer:
    """
    FinBERT-based sentiment analysis for financial text.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize FinBERT analyzer.
        
        Args:
            model_name: Hugging Face model name for FinBERT
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading FinBERT model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Map model labels to our sentiment categories
            self.label_mapping = {
                0: 'positive',  # Bullish/Hawkish
                1: 'negative',  # Bearish/Dovish  
                2: 'neutral'    # Neutral
            }
            
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            self.tokenizer = None
            self.model = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, Union[float, str]]:
        """
        Analyze sentiment of financial text using FinBERT.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if self.model is None or self.tokenizer is None:
            logger.error("FinBERT model not loaded")
            return self._get_default_sentiment()
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            if not processed_text:
                return self._get_default_sentiment()
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Extract results
            probabilities = predictions.cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            
            # Map to our sentiment categories
            sentiment = self._map_to_policy_sentiment(predicted_class)
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'raw_probabilities': probabilities.tolist(),
                'model_prediction': predicted_class,
                'processed_text': processed_text
            }
            
        except Exception as e:
            logger.error(f"Error in FinBERT sentiment analysis: {e}")
            return self._get_default_sentiment()
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for FinBERT analysis.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove extra whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate if too long (FinBERT has 512 token limit)
        if len(text) > 2000:  # Rough character limit
            text = text[:2000] + "..."
        
        return text
    
    def _map_to_policy_sentiment(self, predicted_class: int) -> str:
        """
        Map FinBERT prediction to policy sentiment categories.
        
        Args:
            predicted_class: FinBERT predicted class
            
        Returns:
            Policy sentiment category
        """
        # FinBERT typically predicts: 0=positive, 1=negative, 2=neutral
        # For monetary policy, we interpret:
        # - Positive (bullish) -> Hawkish (tightening)
        # - Negative (bearish) -> Dovish (accommodative)
        # - Neutral -> Neutral
        
        mapping = {
            0: 'hawkish',   # Positive/Bullish -> Hawkish
            1: 'dovish',    # Negative/Bearish -> Dovish
            2: 'neutral'    # Neutral -> Neutral
        }
        
        return mapping.get(predicted_class, 'neutral')
    
    def _get_default_sentiment(self) -> Dict[str, Union[float, str]]:
        """
        Get default sentiment result for error cases.
        """
        return {
            'sentiment': 'neutral',
            'confidence': 0.0,
            'raw_probabilities': [0.33, 0.33, 0.34],
            'model_prediction': 2,
            'processed_text': ''
        }
    
    def analyze_batch(
        self, 
        texts: List[str],
        batch_size: int = 8
    ) -> List[Dict[str, Union[float, str]]]:
        """
        Analyze sentiment for multiple texts in batches.
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing
            
        Returns:
            List of sentiment analysis results
        """
        logger.info(f"Analyzing sentiment for {len(texts)} texts using FinBERT")
        
        if self.model is None or self.tokenizer is None:
            logger.error("FinBERT model not loaded")
            return [self._get_default_sentiment() for _ in texts]
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self._process_batch(batch_texts)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, texts: List[str]) -> List[Dict[str, Union[float, str]]]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of texts in batch
            
        Returns:
            List of sentiment analysis results
        """
        try:
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Tokenize batch
            inputs = self.tokenizer(
                processed_texts,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Process results
            batch_results = []
            probabilities = predictions.cpu().numpy()
            
            for i, prob in enumerate(probabilities):
                predicted_class = np.argmax(prob)
                confidence = float(prob[predicted_class])
                sentiment = self._map_to_policy_sentiment(predicted_class)
                
                batch_results.append({
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'raw_probabilities': prob.tolist(),
                    'model_prediction': predicted_class,
                    'processed_text': processed_texts[i]
                })
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return [self._get_default_sentiment() for _ in texts]
    
    def analyze_policy_events(
        self, 
        events_df: pd.DataFrame,
        text_column: str = 'content'
    ) -> pd.DataFrame:
        """
        Analyze sentiment for policy events DataFrame using FinBERT.
        
        Args:
            events_df: DataFrame with policy events
            text_column: Column containing text to analyze
            
        Returns:
            DataFrame with sentiment analysis results
        """
        logger.info(f"Analyzing sentiment for {len(events_df)} policy events using FinBERT")
        
        if events_df.empty:
            return pd.DataFrame()
        
        # Prepare texts for analysis
        texts = events_df[text_column].fillna('').tolist()
        
        # Analyze sentiment
        sentiment_results = self.analyze_batch(texts)
        
        # Add results to DataFrame
        result_df = events_df.copy()
        
        result_df['sentiment_finbert'] = [r['sentiment'] for r in sentiment_results]
        result_df['sentiment_confidence_finbert'] = [r['confidence'] for r in sentiment_results]
        result_df['sentiment_probabilities_finbert'] = [r['raw_probabilities'] for r in sentiment_results]
        result_df['model_prediction_finbert'] = [r['model_prediction'] for r in sentiment_results]
        
        return result_df
    
    def get_sentiment_summary(self, sentiment_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for FinBERT sentiment analysis.
        
        Args:
            sentiment_df: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        if sentiment_df.empty:
            return {}
        
        summary = {
            'total_events': len(sentiment_df),
            'sentiment_distribution': sentiment_df['sentiment_finbert'].value_counts().to_dict(),
            'average_confidence': sentiment_df['sentiment_confidence_finbert'].mean(),
            'confidence_by_sentiment': sentiment_df.groupby('sentiment_finbert')['sentiment_confidence_finbert'].mean().to_dict()
        }
        
        return summary
    
    def compare_with_llm(
        self, 
        llm_results: pd.DataFrame, 
        finbert_results: pd.DataFrame
    ) -> Dict:
        """
        Compare FinBERT results with LLM results.
        
        Args:
            llm_results: DataFrame with LLM sentiment results
            finbert_results: DataFrame with FinBERT sentiment results
            
        Returns:
            Dictionary with comparison results
        """
        if llm_results.empty or finbert_results.empty:
            return {}
        
        # Merge results
        comparison_df = llm_results.merge(
            finbert_results[['sentiment_finbert', 'sentiment_confidence_finbert']],
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Calculate agreement
        agreement = (comparison_df['sentiment'] == comparison_df['sentiment_finbert']).mean()
        
        # Confidence comparison
        confidence_correlation = comparison_df['sentiment_confidence'].corr(
            comparison_df['sentiment_confidence_finbert']
        )
        
        comparison_results = {
            'total_comparisons': len(comparison_df),
            'agreement_rate': agreement,
            'confidence_correlation': confidence_correlation,
            'llm_sentiment_distribution': comparison_df['sentiment'].value_counts().to_dict(),
            'finbert_sentiment_distribution': comparison_df['sentiment_finbert'].value_counts().to_dict()
        }
        
        return comparison_results
