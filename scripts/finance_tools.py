import pandas as pd
import yfinance as yf
import talib as ta
import numpy as np 

class StockAnalyzer:
    """
    Manages stock data loading and technical indicator calculation.

    Methods
    -------
    load_price() : Loads historical price data into self.df.
    add_indicators() : Computes and adds TA-Lib technical indicators.
    """
    
    def __init__(self, ticker: str, start=None, end=None, interval="1d"):
        """
        Initializes the analyzer by loading the stock's price data.
        
        Parameters
        ----------
        ticker : str
            The stock ticker symbol (e.g., 'AAPL').
        start, end : str, optional
            Date range for data retrieval (e.g., '2023-01-01').
        interval : str, optional
            Data interval (e.g., '1d', '1wk').
        """
        self.ticker = ticker
        self.df = self._load_price(ticker, start, end, interval)

    def _load_price(self, ticker: str, start=None, end=None, interval="1d") -> pd.DataFrame:
        """Loads price data for a given ticker using yfinance."""
        # Note: Added underscore to make this method 'internal'
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
        df.index = pd.to_datetime(df.index)
        
        # Flatten MultiIndex columns if they exist.
        if isinstance(df.columns, pd.MultiIndex):
            # Assumes the second level is the ticker
            df.columns = df.columns.droplevel(1) 
        
        # Drop any rows that yfinance returns with all NaNs (e.g., initial row)
        df.dropna(how='all', inplace=True) 

        return df
    
    def add_returns(self):
       """
        Calculates the daily percentage return and drops the initial NaN row.
        Updates self.df internally and returns the updated DataFrame.
      """
       # Create a copy to perform calculations on
       df = self.df.copy() 
    
       # Calculate daily returns
       df['Return'] = df['Close'].pct_change()
    
       # Drop the first row which will contain NaN after pct_change
       df.dropna(subset=['Return'], inplace=True)
       
       # CRITICAL FIX: Update the internal DataFrame
       self.df = df
       
       return self.df # Return the updated DataFrame

    def add_indicators(self):
        """
        Computes and adds technical indicators (SMA_20, SMA_50, RSI_14, MACD) 
        to the internal DataFrame. Drops rows with NaN values resulting from 
        the indicator lookback periods.
        """
        df = self.df.copy()
        
        # We only use 'Close' for TA-Lib, but converting only once is clean
        close_prices = df["Close"].values.astype(np.float64)
        
        # Check for sufficient data length (SMA_50 is the longest period used)
        if len(close_prices) < 50:
            raise ValueError(
                f"Not enough data (only {len(close_prices)} points) for TA-Lib indicators (SMA_50 requires at least 50)."
            )

        # 1. Simple Moving Average (SMA)
        df["SMA_20"] = ta.SMA(close_prices, timeperiod=20)
        df["SMA_50"] = ta.SMA(close_prices, timeperiod=50)

        # 2. Relative Strength Index (RSI)
        df["RSI_14"] = ta.RSI(close_prices, timeperiod=14)

        # 3. Moving Average Convergence Divergence (MACD)
        macd, macdsignal, macdhist = ta.MACD(
            close_prices, fastperiod=12, slowperiod=26, signalperiod=9
        )
        df["MACD"] = macd
        df["MACD_Signal"] = macdsignal
        df["MACD_Hist"] = macdhist
        
        # List of all new indicator columns
        indicator_cols = ['SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist']
        
        # FIX: Drop only the rows where the new indicator columns have NaNs (due to lookback)
        # This keeps the original data intact if possible, but ensures the calculated indicators are clean.
        self.df = df.dropna(subset=indicator_cols).copy()

        return self.df