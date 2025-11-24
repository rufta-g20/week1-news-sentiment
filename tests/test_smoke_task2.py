"""
End-to-End (E2E) Smoke Test:
Verifies that the main workflow—loading data, calculating finance indicators,
and processing text—runs without crashing and produces expected column outputs.
This simulates a basic end-to-end run of the core functionality.
"""
import pandas as pd
import numpy as np
import pytest
from scripts.finance_tools import StockAnalyzer
from scripts.text_processing import NewsCorpusProcessor

# --- Mock Data ---
# Create a dummy DataFrame for testing text processing
MOCK_NEWS_DATA = pd.DataFrame({
    'headline': [
        "Company A reports huge 20% gain in stock price, major news.",
        "Analyst says buy the dip: oil prices tumble down.",
        "Another headline about markets and money.",
        np.nan,
        "The quick brown fox jumps over the lazy dog."
    ]
})

@pytest.fixture(scope="module")
def stock_analyzer():
    """Fixture to load a small amount of real stock data for testing."""
    # We use a short period to make the test fast and avoid overwhelming yfinance
    # Note: StockAnalyzer handles its own data loading
    try:
        analyzer = StockAnalyzer(ticker='AAPL', start='2024-11-01', end='2024-11-20')
        return analyzer
    except Exception as e:
        pytest.skip(f"Could not load real stock data (yfinance/network issue): {e}")

def test_e2e_workflow(stock_analyzer):
    """
    Tests the sequence: Load Price -> Add Indicators -> Text Processing.
    """
    # 1. Financial Analysis (StockAnalyzer)
    # Check that the data loaded has the expected columns
    assert not stock_analyzer.df.empty
    
    # Check that add_indicators runs without error and adds new columns
    stock_analyzer.add_indicators()
    
    # Check for core indicator columns (SMA_50 requires 50 data points, so we only check smaller ones)
    expected_finance_cols = ['RSI_14', 'MACD', 'MACD_Signal']
    
    # We only check for the presence of the columns, not their values, in a smoke test
    # SMA_50 might not be generated if data is too short, so we'll check RSI/MACD which have shorter periods.
    # The StockAnalyzer code now raises a ValueError if data is too short, which is a good protection.
    for col in expected_finance_cols:
        assert col in stock_analyzer.df.columns, f"Financial Indicator column '{col}' is missing after processing."

    # 2. Text Analysis (NewsCorpusProcessor)
    processor = NewsCorpusProcessor(MOCK_NEWS_DATA['headline'].tolist())
    
    # Check that corpus preparation runs without error and creates outputs
    processor.prepare_corpus(no_below=1) # Use a low no_below for the tiny mock corpus
    assert processor.dictionary is not None
    assert len(processor.corpus) == len(MOCK_NEWS_DATA)
    
    # Check that LDA runs without error and returns topics
    model, topics = processor.lda_topics(num_topics=2, passes=1)
    assert model is not None
    assert len(topics) == 2, "LDA did not return the correct number of topics."
    
    # E2E Check Passed: The entire chain of command executes without exceptions.