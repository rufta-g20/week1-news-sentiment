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

    def add_indicators(self):
        """
        Adds technical analysis indicators (SMA, RSI, MACD) to the internal DataFrame (self.df).
        """
        df = self.df.copy()
        
        # Coerce 'Close' column to numeric to handle any data errors.
        df["Close"] = pd.to_numeric(df["Close"], errors='coerce') 

        # Drop any remaining rows with NaNs.
        df.dropna(inplace=True) 

        # Extract the clean NumPy array for TA-Lib, explicitly cast as float64.
        close_prices = df["Close"].values.astype(np.float64)
        
        # Check for sufficient data length (SMA_50 is the longest period used)
        if len(close_prices) < 50:
            raise ValueError(
                f"Not enough clean data (only {len(close_prices)} points) for TA-Lib indicators (SMA_50 requires at least 50)."
            )

        # ----------------------------------------------------
        # 1. Simple Moving Average (SMA) - Tracks average price
        # ----------------------------------------------------
        df["SMA_20"] = ta.SMA(close_prices, timeperiod=20)
        df["SMA_50"] = ta.SMA(close_prices, timeperiod=50)

        # 2. Relative Strength Index (RSI) - Measures speed and change of price movements
        df["RSI_14"] = ta.RSI(close_prices, timeperiod=14)

        # 3. Moving Average Convergence Divergence (MACD)
        # MACD: 12-period EMA - 26-period EMA
        macd, macdsignal, macdhist = ta.MACD(
            close_prices, fastperiod=12, slowperiod=26, signalperiod=9
        )
        df["MACD"] = macd
        df["MACD_Signal"] = macdsignal
        
        # Update the internal DataFrame
        self.df = df.dropna()
        return self.df