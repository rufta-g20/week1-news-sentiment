
"""
Unit tests for the NewsCorpusProcessor class and utility functions
in scripts/text_processing.py.
"""
import pandas as pd
import pytest
from scripts.text_processing import NewsCorpusProcessor, clean_text, headline_length, publisher_domain

# --- Fixtures and Setup ---

MOCK_TEXT_LIST = [
    "  Breaking News! Apple's stock is up 5% today (NASDAQ:AAPL). ",
    "A major announcement by J.P. Morgan, but is it real?",
    "markets are volatile; buy, sell, or hold?",
    "", # Empty string
    "Just words."
]

@pytest.fixture
def news_processor():
    """Fixture to create a processor instance with mock data."""
    return NewsCorpusProcessor(MOCK_TEXT_LIST)

# --- Tests for Utility Functions ---

def test_clean_text():
    """Test the clean_text function for lowercasing, punctuation, and whitespace."""
    text = "  Breaking News! Apple's stock is up 5% today (NASDAQ:AAPL). "
    expected = "breaking news apple s stock is up today"
    assert clean_text(text) == expected

def test_headline_length():
    """Test the headline_length utility function for character and token counts."""
    df = pd.DataFrame({'headline': MOCK_TEXT_LIST})
    result = headline_length(df, col='headline')
    
    # Check shape
    assert result.shape == (5, 2)
    
    # Check token counts (splits by whitespace)
    assert result['tokens'].iloc[0] == 9
    assert result['tokens'].iloc[2] == 7
    assert result['tokens'].iloc[3] == 0 # Empty string
    
    # Check char counts (includes whitespace, excludes leading/trailing spaces if any)
    assert result['chars'].iloc[0] == 63
    assert result['chars'].iloc[3] == 0

def test_publisher_domain():
    """Test the publisher_domain utility function for email and URL extraction."""
    assert publisher_domain("user@example.com") == "example.com"
    assert publisher_domain("https://www.ft.com/news/main") == "ft.com"
    assert publisher_domain("The Guardian") == "the guardian" # No domain/suffix to extract
    assert publisher_domain(None) == ""
    assert publisher_domain(pd.NA) == ""
    assert publisher_domain("marketwatch") == "marketwatch" # Should return the domain if no suffix
    assert publisher_domain("bloomberg.com") == "bloomberg.com"

# --- Tests for NewsCorpusProcessor Class ---

def test_processor_initialization(news_processor):
    """Test that the class initializes correctly."""
    assert len(news_processor.texts) == len(MOCK_TEXT_LIST)
    assert news_processor.dictionary is None
    assert news_processor.corpus is None

def test_prepare_corpus(news_processor):
    """Test that prepare_corpus tokenizes, cleans, and generates dictionary/corpus."""
    news_processor.prepare_corpus(no_below=1) # Low threshold for tiny data
    
    # Check dictionary size (should be small, excluding stop words and short words)
    assert news_processor.dictionary is not None
    # 'breaking', 'news', 'apple', 'stock', 'today', 'nasdaq', 'aapl' (7)
    # 'major', 'announcement', 'morgan', 'real' (4)
    # 'markets', 'volatile', 'buy', 'sell', 'hold' (5)
    # 'words' (1)
    # Total unique words > 10
    assert len(news_processor.dictionary) > 10
    
    # Check corpus size
    assert len(news_processor.corpus) == len(MOCK_TEXT_LIST)
    
    # The last text "Just words." becomes "words" (stop words removed).
    word_id = news_processor.dictionary.token2id.get('words')
    # If 'words' is in the dictionary, the last document should have one token.
    if word_id is not None:
        assert news_processor.corpus[-1] == [(word_id, 1)]

def test_lda_topics_before_prepare(news_processor):
    """Test that lda_topics raises error if prepare_corpus was not run first."""
    with pytest.raises(ValueError, match="Corpus and dictionary must be prepared"):
        news_processor.lda_topics()

def test_lda_topics_execution(news_processor):

    """Test that lda_topics runs successfully after preparing the corpus."""
    news_processor.prepare_corpus(no_below=1) 
    
    model, topics = news_processor.lda_topics(num_topics=2, passes=1)
    
    # Check model and topic output
    assert model is not None
    assert len(topics) == 2
    assert "Topic 0" in topics[0]
    assert "Topic 1" in topics[1]