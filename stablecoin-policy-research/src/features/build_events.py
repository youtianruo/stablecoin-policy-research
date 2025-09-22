"""
Event calendar builder for policy events and analysis windows.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class EventBuilder:
    """
    Builds event calendars and analysis windows for policy events.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize event builder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.event_window_pre = config.get('policy_events', {}).get('event_window', {}).get('pre', 5)
        self.event_window_post = config.get('policy_events', {}).get('event_window', {}).get('post', 5)
        self.estimation_window = config.get('policy_events', {}).get('estimation_window', 250)
    
    def build_event_calendar(self, policy_events: pd.DataFrame) -> pd.DataFrame:
        """
        Build comprehensive event calendar with analysis windows.
        
        Args:
            policy_events: DataFrame with policy events
            
        Returns:
            DataFrame with event calendar including analysis windows
        """
        logger.info("Building event calendar")
        
        if policy_events.empty:
            logger.warning("No policy events provided")
            return pd.DataFrame()
        
        event_calendar = []
        
        for idx, event in policy_events.iterrows():
            event_date = event['date']
            
            # Create event window
            event_window = self._create_event_window(event_date)
            
            # Create estimation window
            estimation_window = self._create_estimation_window(event_date)
            
            # Add to calendar
            calendar_entry = {
                'event_id': idx,
                'event_date': event_date,
                'event_type': event['event_type'],
                'title': event.get('title', ''),
                'content': event.get('content', ''),
                'url': event.get('url', ''),
                'speaker': event.get('speaker', ''),
                'event_window_start': event_window[0],
                'event_window_end': event_window[1],
                'estimation_window_start': estimation_window[0],
                'estimation_window_end': estimation_window[1],
                'pre_event_days': self.event_window_pre,
                'post_event_days': self.event_window_post,
                'estimation_days': self.estimation_window
            }
            
            event_calendar.append(calendar_entry)
        
        calendar_df = pd.DataFrame(event_calendar)
        calendar_df = calendar_df.sort_values('event_date').reset_index(drop=True)
        
        logger.info(f"Built event calendar with {len(calendar_df)} events")
        return calendar_df
    
    def _create_event_window(self, event_date: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Create event window around policy event.
        
        Args:
            event_date: Date of the policy event
            
        Returns:
            Tuple of (start_date, end_date) for event window
        """
        # Convert to business days
        start_date = event_date - pd.Timedelta(days=self.event_window_pre)
        end_date = event_date + pd.Timedelta(days=self.event_window_post)
        
        # Adjust to business days
        start_date = pd.bdate_range(start_date, start_date)[0]
        end_date = pd.bdate_range(end_date, end_date)[0]
        
        return start_date, end_date
    
    def _create_estimation_window(self, event_date: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Create estimation window for normal returns calculation.
        
        Args:
            event_date: Date of the policy event
            
        Returns:
            Tuple of (start_date, end_date) for estimation window
        """
        # Estimation window ends before event window starts
        end_date = event_date - pd.Timedelta(days=self.event_window_pre + 1)
        start_date = end_date - pd.Timedelta(days=self.estimation_window)
        
        # Adjust to business days
        start_date = pd.bdate_range(start_date, start_date)[0]
        end_date = pd.bdate_range(end_date, end_date)[0]
        
        return start_date, end_date
    
    def categorize_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize events by type and importance.
        
        Args:
            events_df: DataFrame with policy events
            
        Returns:
            DataFrame with categorized events
        """
        logger.info("Categorizing events")
        
        categorized_events = events_df.copy()
        
        # Add event importance score
        categorized_events['importance'] = self._calculate_event_importance(categorized_events)
        
        # Add event category
        categorized_events['category'] = self._categorize_event_type(categorized_events)
        
        # Add market impact expectation
        categorized_events['expected_impact'] = self._estimate_market_impact(categorized_events)
        
        return categorized_events
    
    def _calculate_event_importance(self, events_df: pd.DataFrame) -> pd.Series:
        """
        Calculate importance score for each event.
        
        Args:
            events_df: DataFrame with policy events
            
        Returns:
            Series with importance scores
        """
        importance_scores = []
        
        for _, event in events_df.iterrows():
            score = 0
            
            # Base score by event type
            event_type_scores = {
                'fomc_meeting': 10,
                'fomc_minutes': 8,
                'rate_decision': 9,
                'fed_speech': 5,
                'qt_announcement': 7,
                'testimony': 6
            }
            
            score += event_type_scores.get(event['event_type'], 3)
            
            # Adjust based on content length (proxy for detail)
            if 'content' in event and event['content']:
                content_length = len(event['content'])
                if content_length > 5000:
                    score += 2
                elif content_length > 2000:
                    score += 1
            
            # Adjust based on speaker (if available)
            if 'speaker' in event and event['speaker']:
                important_speakers = ['Jerome Powell', 'John Williams', 'Lael Brainard']
                if any(speaker in event['speaker'] for speaker in important_speakers):
                    score += 2
            
            importance_scores.append(min(score, 10))  # Cap at 10
        
        return pd.Series(importance_scores, index=events_df.index)
    
    def _categorize_event_type(self, events_df: pd.DataFrame) -> pd.Series:
        """
        Categorize events into broader categories.
        
        Args:
            events_df: DataFrame with policy events
            
        Returns:
            Series with event categories
        """
        categories = []
        
        for _, event in events_df.iterrows():
            event_type = event['event_type']
            
            if event_type in ['fomc_meeting', 'fomc_minutes']:
                categories.append('fomc_communication')
            elif event_type in ['rate_decision', 'qt_announcement']:
                categories.append('policy_action')
            elif event_type in ['fed_speech', 'testimony']:
                categories.append('communication')
            else:
                categories.append('other')
        
        return pd.Series(categories, index=events_df.index)
    
    def _estimate_market_impact(self, events_df: pd.DataFrame) -> pd.Series:
        """
        Estimate expected market impact of events.
        
        Args:
            events_df: DataFrame with policy events
            
        Returns:
            Series with impact estimates
        """
        impact_scores = []
        
        for _, event in events_df.iterrows():
            score = 0
            
            # Base impact by event type
            event_type_impact = {
                'fomc_meeting': 8,
                'fomc_minutes': 6,
                'rate_decision': 9,
                'fed_speech': 4,
                'qt_announcement': 7,
                'testimony': 5
            }
            
            score += event_type_impact.get(event['event_type'], 2)
            
            # Adjust based on importance
            if 'importance' in event:
                score += event['importance'] * 0.3
            
            impact_scores.append(min(score, 10))  # Cap at 10
        
        return pd.Series(impact_scores, index=events_df.index)
    
    def filter_events_by_importance(
        self, 
        events_df: pd.DataFrame, 
        min_importance: float = 5.0
    ) -> pd.DataFrame:
        """
        Filter events by minimum importance score.
        
        Args:
            events_df: DataFrame with categorized events
            min_importance: Minimum importance score
            
        Returns:
            Filtered DataFrame
        """
        if 'importance' not in events_df.columns:
            logger.warning("Importance column not found, returning all events")
            return events_df
        
        filtered_events = events_df[events_df['importance'] >= min_importance].copy()
        
        logger.info(f"Filtered to {len(filtered_events)} events with importance >= {min_importance}")
        return filtered_events
    
    def get_event_summary(self, events_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for events.
        
        Args:
            events_df: DataFrame with events
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_events': len(events_df),
            'event_types': events_df['event_type'].value_counts().to_dict(),
            'date_range': {
                'start': events_df['date'].min(),
                'end': events_df['date'].max()
            }
        }
        
        if 'importance' in events_df.columns:
            summary['importance_stats'] = {
                'mean': events_df['importance'].mean(),
                'std': events_df['importance'].std(),
                'min': events_df['importance'].min(),
                'max': events_df['importance'].max()
            }
        
        if 'category' in events_df.columns:
            summary['categories'] = events_df['category'].value_counts().to_dict()
        
        return summary
