"""
Macroeconomic data fetcher using FRED API.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from fredapi import Fred

logger = logging.getLogger(__name__)


class MacroDataFetcher:
    """
    Fetches macroeconomic data from FRED (Federal Reserve Economic Data).
    """
    
    def __init__(self, api_key: str):
        """
        Initialize FRED data fetcher.
        
        Args:
            api_key: FRED API key
        """
        self.api_key = api_key
        self.fred = Fred(api_key=api_key)
    
    def get_fed_funds_rate(
        self, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime]
    ) -> pd.Series:
        """
        Fetch Federal Funds rate data.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Series with Federal Funds rate
        """
        logger.info("Fetching Federal Funds rate")
        
        try:
            data = self.fred.get_series(
                'FEDFUNDS',
                start_date,
                end_date
            )
            return data
        except Exception as e:
            logger.error(f"Error fetching Federal Funds rate: {e}")
            return pd.Series()
    
    def get_treasury_rates(
        self, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime],
        maturities: List[str] = None
    ) -> pd.DataFrame:
        """
        Fetch Treasury rates for different maturities.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            maturities: List of maturity codes (e.g., ['DGS3MO', 'DGS1', 'DGS2', 'DGS10'])
            
        Returns:
            DataFrame with Treasury rates
        """
        if maturities is None:
            maturities = ['DGS3MO', 'DGS1', 'DGS2', 'DGS5', 'DGS10', 'DGS30']
        
        logger.info(f"Fetching Treasury rates for maturities: {maturities}")
        
        rates_data = {}
        
        for maturity in maturities:
            try:
                data = self.fred.get_series(
                    maturity,
                    start_date,
                    end_date
                )
                rates_data[maturity] = data
            except Exception as e:
                logger.error(f"Error fetching Treasury rate {maturity}: {e}")
                continue
        
        if rates_data:
            return pd.DataFrame(rates_data)
        else:
            return pd.DataFrame()
    
    def get_inflation_data(
        self, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        Fetch inflation-related data.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with inflation data
        """
        logger.info("Fetching inflation data")
        
        inflation_series = {
            'CPI': 'CPIAUCSL',  # Consumer Price Index
            'Core_CPI': 'CPILFESL',  # Core CPI
            'PCE': 'PCEPI',  # Personal Consumption Expenditures
            'Core_PCE': 'PCEPILFE',  # Core PCE
            'Breakeven_10Y': 'T10YIE',  # 10-Year Breakeven Inflation Rate
            'Breakeven_5Y': 'T5YIE'  # 5-Year Breakeven Inflation Rate
        }
        
        inflation_data = {}
        
        for name, series_id in inflation_series.items():
            try:
                data = self.fred.get_series(
                    series_id,
                    start_date,
                    end_date
                )
                inflation_data[name] = data
            except Exception as e:
                logger.error(f"Error fetching inflation data {name}: {e}")
                continue
        
        if inflation_data:
            return pd.DataFrame(inflation_data)
        else:
            return pd.DataFrame()
    
    def get_employment_data(
        self, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        Fetch employment-related data.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with employment data
        """
        logger.info("Fetching employment data")
        
        employment_series = {
            'Unemployment_Rate': 'UNRATE',
            'Nonfarm_Payrolls': 'PAYEMS',
            'Labor_Force_Participation': 'CIVPART',
            'Average_Hourly_Earnings': 'AHETPI'
        }
        
        employment_data = {}
        
        for name, series_id in employment_series.items():
            try:
                data = self.fred.get_series(
                    series_id,
                    start_date,
                    end_date
                )
                employment_data[name] = data
            except Exception as e:
                logger.error(f"Error fetching employment data {name}: {e}")
                continue
        
        if employment_data:
            return pd.DataFrame(employment_data)
        else:
            return pd.DataFrame()
    
    def get_gdp_data(
        self, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        Fetch GDP-related data.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with GDP data
        """
        logger.info("Fetching GDP data")
        
        gdp_series = {
            'GDP': 'GDP',
            'Real_GDP': 'GDPC1',
            'GDP_Growth_Rate': 'A191RL1Q225SBEA',
            'Personal_Consumption': 'PCE',
            'Gross_Private_Investment': 'GPDI',
            'Government_Spending': 'GCE'
        }
        
        gdp_data = {}
        
        for name, series_id in gdp_series.items():
            try:
                data = self.fred.get_series(
                    series_id,
                    start_date,
                    end_date
                )
                gdp_data[name] = data
            except Exception as e:
                logger.error(f"Error fetching GDP data {name}: {e}")
                continue
        
        if gdp_data:
            return pd.DataFrame(gdp_data)
        else:
            return pd.DataFrame()
    
    def get_financial_indicators(
        self, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        Fetch financial market indicators.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with financial indicators
        """
        logger.info("Fetching financial indicators")
        
        financial_series = {
            'VIX': 'VIXCLS',  # Volatility Index
            'DXY': 'DTWEXBGS',  # Dollar Index
            'Credit_Spread': 'BAMLH0A0HYM2',  # High Yield Spread
            'Term_Spread': 'T10Y2Y',  # 10Y-2Y Treasury Spread
            'Mortgage_Rate': 'MORTGAGE30US'  # 30-Year Mortgage Rate
        }
        
        financial_data = {}
        
        for name, series_id in financial_series.items():
            try:
                data = self.fred.get_series(
                    series_id,
                    start_date,
                    end_date
                )
                financial_data[name] = data
            except Exception as e:
                logger.error(f"Error fetching financial indicator {name}: {e}")
                continue
        
        if financial_data:
            return pd.DataFrame(financial_data)
        else:
            return pd.DataFrame()
    
    def get_all_macro_data(
        self, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all macroeconomic data.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary with all macro data
        """
        logger.info("Fetching all macroeconomic data")
        
        macro_data = {}
        
        # Fetch different categories of data
        macro_data['fed_funds'] = self.get_fed_funds_rate(start_date, end_date)
        macro_data['treasury_rates'] = self.get_treasury_rates(start_date, end_date)
        macro_data['inflation'] = self.get_inflation_data(start_date, end_date)
        macro_data['employment'] = self.get_employment_data(start_date, end_date)
        macro_data['gdp'] = self.get_gdp_data(start_date, end_date)
        macro_data['financial'] = self.get_financial_indicators(start_date, end_date)
        
        return macro_data
    
    def get_series_info(self, series_id: str) -> Dict:
        """
        Get information about a FRED series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Dictionary with series information
        """
        try:
            info = self.fred.get_series_info(series_id)
            return info
        except Exception as e:
            logger.error(f"Error fetching series info for {series_id}: {e}")
            return {}
    
    def search_series(self, search_text: str, limit: int = 20) -> pd.DataFrame:
        """
        Search for FRED series.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            DataFrame with search results
        """
        try:
            results = self.fred.search(search_text, limit=limit)
            return results
        except Exception as e:
            logger.error(f"Error searching FRED series: {e}")
            return pd.DataFrame()
