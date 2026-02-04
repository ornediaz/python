from typing import cast
import pandas as pd
from datetime import date

from schedule import Schedule

class MarketDataError(Exception):
    """Custom exception for MarketData errors"""
    pass

class MarketData:
    """
    A class to load and query market data from a CSV file.
    
    The CSV file should have columns: date, ticker, close
    """
    
    def __init__(self, filename: str):
        """
        Initialize MarketData with a CSV file.
        
        Args:
            filename (str): Path to the CSV file containing market data
        """
        self._data = self._load_data(filename)
    
    def _load_data(self, filename: str) -> pd.DataFrame:
        """Load from a CSV file."""
        try:
            df = pd.read_csv(filename)
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Set multi-index for fast lookups
            df = df.set_index(['date', 'ticker'])
            
            return df
            
        except FileNotFoundError:
            raise MarketDataError(f"File not found: {filename}")
        except Exception as e:
            raise MarketDataError(f"Error loading data from {filename}: {e}")
    
    def get(self, date: date, ticker: str) -> float:
        """
        Get the closing price for a specific date and ticker.
        
        Args:
            date: Date to query
            ticker: Ticker symbol (e.g., 'SPX', 'SX5E', 'HSI')
        
        Returns:
            float: The closing price for the given date and ticker
            
        Raises:
            MarketDataError: If the requested date/ticker combination is not found
        """
        try:
            return cast(float, self._data.loc[(pd.to_datetime(date), ticker), 'close'])
        except KeyError:
            raise MarketDataError(f"No data for '{ticker}' on {date}.")
        
    def get_calendar(self) -> Schedule:
        """
        Get all available dates in the dataset.
        
        Returns:
            Schedule: Sorted list of all unique dates in the dataset
        """
        return Schedule(self._data.index.get_level_values('date'))
