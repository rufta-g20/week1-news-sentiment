import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import sys, os
from datetime import date

# Ensure the scripts directory is on the path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(file), '..', 'scripts')))

from finance_tools import StockAnalyzer

# --- Fixtures / Mock Data ---

# Create a deterministic mock DataFrame for yfinance download
# We need at least 50 data points for SMA_50 to calculate, but we'll use 60 for buffer.
@pytest.fixture
def mock_stock_data():
    """Generates a mock DataFrame suitable for testing TA-Lib indicators."""
    num_days = 60
    dates = pd.date_range(start='2024-01-01', periods=num_days, freq='D')
    # Create simple, non-random price data for predictable indicator results (Close, Open, High, Low)
    data = {
        'Open': np.linspace(100, 150, num_days),
        'High': np.linspace(101, 151, num_days),
        'Low': np.linspace(99, 149, num_days),
        'Close': np.linspace(100.5, 150.5, num_days)
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

# Helper to mock the yfinance download function
def mock_download(*args, **kwargs):
    """Mocks yfinance.download to return a fixed, predictable DataFrame."""
    return mock_stock_data()


# --- Tests for StockAnalyzer ---

@patch('yfinance.download', side_effect=mock_download)
def test_stock_analyzer_init(mock_download_func, mock_stock_data):
    """Test that the StockAnalyzer initializes and loads data correctly."""
    analyzer = StockAnalyzer(ticker='TEST')
    
    # Check that yfinance.download was called
    mock_download_func.assert_called_once()
    
    # Check that the loaded DataFrame is stored and matches the mock data structure
    assert isinstance(analyzer.df, pd.DataFrame)
    assert analyzer.df.shape == mock_stock_data.shape
    assert 'Close' in analyzer.df.columns
    assert 'Date' == analyzer.df.index.name
    
@patch('yfinance.download', side_effect=mock_download)
def test_add_indicators_creates_columns(mock_download_func):
    """Test that add_indicators adds all required TA-Lib columns."""
    analyzer = StockAnalyzer(ticker='TEST')
    analyzer.add_indicators()
    
    # Expected columns from finance_tools.py
    expected_cols = ['SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Upper_BB', 'Middle_BB', 'Lower_BB']
    
    for col in expected_cols:
        assert col in analyzer.df.columns, f"Missing expected indicator column: {col}"
    
@patch('yfinance.download', side_effect=mock_download)
def test_add_indicators_initial_nan_handling(mock_download_func):
    """Test that TA-Lib correctly handles initial NaN values for indicators based on their period."""
    analyzer = StockAnalyzer(ticker='TEST')
    analyzer.add_indicators()
    df = analyzer.df
    
    # SMA_20 requires 20 periods, so index 19 should be NaN, index 20 should be a number.
    assert pd.isna(df['SMA_20'].iloc[19])
    assert pd.notna(df['SMA_20'].iloc[20])
    
    # SMA_50 requires 50 periods, so index 49 should be NaN, index 50 should be a number.
    assert pd.isna(df['SMA_50'].iloc[49])
    assert pd.notna(df['SMA_50'].iloc[50])
    
    # The length of the dataframe after dropna in add_indicators should be 60 - (50 - 1) = 11 rows of calculated data (but it actually drops NaNs AFTER calculation, so length should be based on initial size - NaNs)
    # The TA-Lib functions return Series where the first N-1 rows are NaN, where N is the longest period (50 for SMA_50)
    # Since the internal data cleaning in StockAnalyzer.add_indicators() drops NaNs, the final length should be reduced.
    # Total rows: 60. Longest lookback: 50. Number of NaNs: 49.

# After dropna(inplace=True), the final DataFrame should have 60 - 49 = 11 rows.
    # Note: I am assuming the internal logic in StockAnalyzer.add_indicators() is updated to handle dropna after calculation.
    # The snippet shows: df.dropna(inplace=True) is called before calculation, which is incorrect as TA-Lib outputs will reintroduce NaNs.
    # I need to assume the logic is: load -> calculate -> drop NaNs from indicators.
    
    # Let's check the number of non-NaN values for SMA_50:
    non_nan_count = df['SMA_50'].count()
    # It should be 60 total rows - 49 initial NaNs = 11 calculated values.
    assert non_nan_count == 11, f"Expected 11 non-NaN values for SMA_50, got {non_nan_count}"

@patch('yfinance.download', side_effect=mock_download)
def test_add_indicators_insufficient_data():
    """Test that add_indicators raises ValueError when data is insufficient (e.g., < 50 rows)."""
    # Override mock data to return only 40 rows
    short_data = mock_stock_data()[:40]

    with patch('yfinance.download', return_value=short_data):
        analyzer = StockAnalyzer(ticker='SHORT')
        # The add_indicators method is expected to raise a ValueError due to the check in finance_tools.py
        with pytest.raises(ValueError) as excinfo:
            analyzer.add_indicators()
            
        assert "Not enough clean data (only 40 points) for TA-Lib indicators" in str(excinfo.value)

@patch('yfinance.download', side_effect=mock_download)
def test_add_returns(mock_download_func, mock_stock_data):
    """Test that add_returns calculates the daily return and drops the initial NaN."""
    analyzer = StockAnalyzer(ticker='TEST')
    processed_df = analyzer.add_returns()
    
    # Check for the new column
    assert 'Return' in processed_df.columns
    # Check that the first row (initial NaN) was dropped
    assert len(processed_df) == len(mock_stock_data) - 1
    # Check that the 'Return' column contains no NaNs
    assert processed_df['Return'].isnull().sum() == 0

@patch('yfinance.download', side_effect=mock_download)
def test_add_indicators_result_size(mock_download_func, mock_stock_data):
    """Test that add_indicators results in a clean DataFrame with the correct number of rows (after dropping TA-Lib NaNs)."""
    analyzer = StockAnalyzer(ticker='TEST')
    processed_df = analyzer.add_indicators()
    
    # Initial mock data has 60 rows.
    # SMA_50 lookback is 49 days, creating 49 NaNs.
    # Final clean DataFrame size should be 60 - 49 = 11 rows.
    expected_size = len(mock_stock_data) - 49 
    assert len(processed_df) == expected_size, f"Expected {expected_size} rows, got {len(processed_df)}"
    
    # Check that the 'SMA_50' column contains no NaNs
    assert processed_df['SMA_50'].isnull().sum() == 0