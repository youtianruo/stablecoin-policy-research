"""
Federal Reserve data fetcher for FOMC minutes, speeches, and policy events.
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import time
import logging
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)


class FedDataFetcher:
    """
    Fetches Federal Reserve data including FOMC minutes, speeches, and policy events.
    """
    
    def __init__(self):
        """Initialize Fed data fetcher."""
        self.base_urls = {
            'fomc_minutes': 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm',
            'speeches': 'https://www.federalreserve.gov/newsevents/speech/',
            'press_releases': 'https://www.federalreserve.gov/newsevents/pressreleases/',
            'testimonies': 'https://www.federalreserve.gov/newsevents/testimony/'
        }
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_fomc_minutes(self, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Fetch FOMC minutes from Federal Reserve website.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with FOMC minutes data
        """
        logger.info("Fetching FOMC minutes")
        
        # Convert dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        minutes_data = []
        
        # Get FOMC calendar page
        response = self.session.get(self.base_urls['fomc_minutes'])
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find FOMC meeting links
        meeting_links = soup.find_all('a', href=re.compile(r'/monetarypolicy/fomcminutes\d+\.htm'))
        
        for link in meeting_links:
            try:
                meeting_date_str = link.text.strip()
                meeting_date = pd.to_datetime(meeting_date_str)
                
                # Check if within date range
                if start_date <= meeting_date <= end_date:
                    minutes_url = f"https://www.federalreserve.gov{link['href']}"
                    
                    # Fetch minutes content
                    minutes_content = self._fetch_minutes_content(minutes_url)
                    
                    minutes_data.append({
                        'date': meeting_date,
                        'event_type': 'fomc_minutes',
                        'url': minutes_url,
                        'content': minutes_content,
                        'title': f"FOMC Minutes - {meeting_date_str}"
                    })
                    
                    time.sleep(1)  # Rate limiting
                    
            except Exception as e:
                logger.error(f"Error processing FOMC minutes link: {e}")
                continue
        
        return pd.DataFrame(minutes_data)
    
    def _fetch_minutes_content(self, url: str) -> str:
        """
        Fetch content from FOMC minutes URL.
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            content_div = soup.find('div', class_='col-xs-12 col-sm-8 col-md-8')
            if content_div:
                # Remove script and style elements
                for script in content_div(["script", "style"]):
                    script.decompose()
                
                text = content_div.get_text()
                # Clean up text
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error fetching minutes content from {url}: {e}")
            return ""
    
    def get_fed_speeches(self, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Fetch Federal Reserve speeches.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with speeches data
        """
        logger.info("Fetching Fed speeches")
        
        # Convert dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        speeches_data = []
        
        # Get speeches page
        response = self.session.get(self.base_urls['speeches'])
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find speech links
        speech_links = soup.find_all('a', href=re.compile(r'/newsevents/speech/\d+'))
        
        for link in speech_links:
            try:
                # Extract date from link text or nearby elements
                date_element = link.find_previous('time') or link.find_next('time')
                if date_element:
                    date_str = date_element.get_text().strip()
                    speech_date = pd.to_datetime(date_str)
                    
                    # Check if within date range
                    if start_date <= speech_date <= end_date:
                        speech_url = f"https://www.federalreserve.gov{link['href']}"
                        
                        # Fetch speech content
                        speech_content = self._fetch_speech_content(speech_url)
                        
                        speeches_data.append({
                            'date': speech_date,
                            'event_type': 'fed_speech',
                            'url': speech_url,
                            'content': speech_content,
                            'title': link.get_text().strip(),
                            'speaker': self._extract_speaker(link)
                        })
                        
                        time.sleep(1)  # Rate limiting
                        
            except Exception as e:
                logger.error(f"Error processing speech link: {e}")
                continue
        
        return pd.DataFrame(speeches_data)
    
    def _fetch_speech_content(self, url: str) -> str:
        """
        Fetch content from speech URL.
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            content_div = soup.find('div', class_='col-xs-12 col-sm-8 col-md-8')
            if content_div:
                # Remove script and style elements
                for script in content_div(["script", "style"]):
                    script.decompose()
                
                text = content_div.get_text()
                # Clean up text
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error fetching speech content from {url}: {e}")
            return ""
    
    def _extract_speaker(self, link_element) -> str:
        """
        Extract speaker name from speech link element.
        """
        try:
            # Look for speaker name in nearby elements
            speaker_element = link_element.find_previous('h4') or link_element.find_next('h4')
            if speaker_element:
                return speaker_element.get_text().strip()
            else:
                return "Unknown"
        except:
            return "Unknown"
    
    def get_policy_events(self, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Fetch all policy-related events (minutes, speeches, press releases).
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with all policy events
        """
        logger.info("Fetching all policy events")
        
        # Get different types of events
        minutes_df = self.get_fomc_minutes(start_date, end_date)
        speeches_df = self.get_fed_speeches(start_date, end_date)
        
        # Combine all events
        all_events = []
        
        if not minutes_df.empty:
            all_events.append(minutes_df)
        
        if not speeches_df.empty:
            all_events.append(speeches_df)
        
        if all_events:
            combined_df = pd.concat(all_events, ignore_index=True)
            combined_df = combined_df.sort_values('date').reset_index(drop=True)
            return combined_df
        else:
            return pd.DataFrame()
    
    def get_rate_decisions(self, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Fetch Federal Funds rate decisions.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with rate decisions
        """
        logger.info("Fetching rate decisions")
        
        # This would typically come from FRED API or Fed website
        # For now, return empty DataFrame
        # In practice, you'd fetch this from FRED API using the FEDFUNDS series
        
        return pd.DataFrame()
    
    def clean_text_content(self, text: str) -> str:
        """
        Clean and preprocess text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        
        # Remove very short words
        words = text.split()
        words = [word for word in words if len(word) > 2]
        
        return ' '.join(words).strip()
