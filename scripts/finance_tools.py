"""Finance helpers: load price via yfinance and compute indicators via TA-Lib."""
import pandas as pd
import yfinance as yf
import talib as ta
import numpy as np 

def load_price(ticker: str, start=None, end=None, interval="1d"):
    """Loads price data for a given ticker using yfinance."""
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    df.index = pd.to_datetime(df.index)
    
    # Flatten MultiIndex columns if they exist.
    # This ensures df["Close"] returns a single Series, not a DataFrame.
    if isinstance(df.columns, pd.MultiIndex):
        # We assume the second level is the ticker (e.g., ('Close', 'AAPL'))
        df.columns = df.columns.droplevel(1) 
    
    # Drop any rows that yfinance returns with all NaNs (e.g., initial row)
    df.dropna(how='all', inplace=True) 

    return df

def add_talib_indicators(df: pd.DataFrame):
    """Adds technical analysis indicators to the DataFrame using TA-Lib."""
    df = df.copy()
    
    # Coerce 'Close' column to numeric.
    # This safely converts any rogue non-numeric values (like text) to NaN.
    df["Close"] = pd.to_numeric(df["Close"], errors='coerce') 

    # Drop any remaining rows with NaNs in the data columns.
    df.dropna(inplace=True) 

    # Extract the clean NumPy array, explicitly cast as float64, for TA-Lib
    close_prices = df["Close"].values.astype(np.float64)
    
    # Check for sufficient data length (SMA_50 is the longest period used)
    if len(close_prices) < 50:
        raise ValueError(f"Not enough clean data (only {len(close_prices)} points) for TA-Lib indicators (SMA_50 requires at least 50).")

    # SMA - Use the clean NumPy array
    df["SMA_20"] = ta.SMA(close_prices, timeperiod=20)
    df["SMA_50"] = ta.SMA(close_prices, timeperiod=50)

    # RSI
    df["RSI_14"] = ta.RSI(close_prices, timeperiod=14)

    # MACD
    macd, macd_signal, macd_hist = ta.MACD(
        close_prices, fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["MACD"] = macd
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist
    
    # Drop the leading NaN rows that result from indicator calculations
    # (These are the first 49 rows which lack a 50-day SMA, etc.)
    df.dropna(inplace=True)

    return df