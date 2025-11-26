"""
Reusable text preprocessing and topic-modeling helpers.
"""
import re
import pandas as pd
import numpy as np
from gensim import corpora, models
# VADER is much faster than TextBlob for lexicon-based analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import List, Optional, Tuple

# Hardcoded Stopwords (bypasses NLTK LookupError)
STOP = set((
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
    'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', 
    "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', 
    "wouldn't", 'com', 'org', 'www'
))


def clean_text(text: str) -> str:
    """
    Cleans text by converting to lowercase, removing specific patterns, 
    punctuation, numbers, and extra whitespace.
    """
    if pd.isna(text):
        return ""
        
    text = str(text).lower()
    
    # 1. REMOVE TICKET SYMBOLS: E.g., $AAPL, (NASDAQ:TSLA)
    # This prevents noise from financial symbols in topic modeling.
    text = re.sub(r'\(\w+:\w+\)', ' ', text) # Removes (NASDAQ:AAPL)
    text = re.sub(r'\$\w+', ' ', text)        # Removes $AAPL
    
    # 2. Remove punctuation, numbers, and non-alphanumeric symbols
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 3. Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def headline_length(df: pd.DataFrame, col='headline') -> pd.DataFrame:
    """
    Return length (chars and tokens) of headline column.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the text data.
    col : str, optional
        Name of the column containing the headlines, by default 'headline'.

    Returns
    -------
    pd.DataFrame
        DataFrame with two columns: 'chars' and 'tokens'.
    
    Examples
    --------
    >>> data = pd.DataFrame({'headline': ['Hello world.', 'Short']})
    >>> lengths = headline_length(data)
    >>> lengths['tokens'].tolist()
    [2, 1]
    """
    s = df[col].fillna("").astype(str)
    chars = s.str.len()
    tokens = s.apply(lambda t: len(t.split()))
    return pd.DataFrame({'chars': chars, 'tokens': tokens})


def publisher_domain(publisher: str) -> str:
    """
    Extract domain-like part if email-like or url-like publisher is given.
    
    Examples
    --------
    >>> publisher_domain("user@example.com")
    'example.com'
    >>> publisher_domain("www.the-financial-times.co.uk/news")
    'the-financial-times.co.uk'
    """
    import tldextract
    if pd.isna(publisher):
        return ""
    p = str(publisher).strip()
    # if email-like
    if "@" in p:
        return p.split("@")[-1].lower()
    # try tldextract
    ex = tldextract.extract(p)
    if ex.domain:
        return f"{ex.domain}.{ex.suffix}" if ex.suffix else ex.domain
    return p.lower()


class NewsCorpusProcessor:
    """
    Manages the processing of a news corpus, including text cleaning,
    sentiment analysis, and preparation for LDA topic modeling.
    Attributes
    ----------
    texts : list of str
        The original list of headlines/texts.
    dictionary : Optional[corpora.Dictionary]
        Gensim dictionary created after preparation.
    corpus : Optional[list of list of tuple]
        Gensim corpus (Bag-of-Words representation) after preparation.
    texts_tok : Optional[list of list of str]
        Tokenized and cleaned texts.

    Examples
    --------
    >>> data = pd.Series(['US stocks rise on good news.', 'The market is fearful.'])
    >>> processor = NewsCorpusProcessor(data)
    >>> sentiment_df = processor.calculate_sentiment()
    >>> sentiment_df.head(1)
       polarity  subjectivity
    0      0.70      0.600000
    """
    def __init__(self, texts: Optional[List[str]] = None):
        """Initializes the processor and stores the text list."""
        # Store the raw text list
        self.texts = texts
        
        # Attributes for LDA
        self.dictionary = None
        self.corpus = None

    def prepare_corpus(self, 
                       no_below: int = 5, 
                       no_above: float = 0.9, 
                       keep_n: int = 10000) -> None:
        """
        Tokenizes texts, creates a Gensim Dictionary, and generates a Corpus.
        It uses the texts stored in self.texts.
        """
        # Ensure texts are available
        if self.texts is None:
            raise ValueError("Text corpus (self.texts) is empty. Initialize the class with a list of texts.")
        

        texts_tok = [[w for w in clean_text(t).split() if w not in STOP and len(w) > 2] for t in self.texts]
        dictionary = corpora.Dictionary(texts_tok)
        dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        corpus = [dictionary.doc2bow(text) for text in texts_tok]
        
        self.dictionary = dictionary
        self.corpus = corpus
        self.texts_tok = texts_tok

    def lda_topics(self, num_topics: int = 6, passes: int = 6) -> Tuple[models.LdaModel, List[str]]:
        """
        Trains an LDA model using the stored dictionary and corpus.

        Parameters
        ----------
        num_topics : int
            The number of topics to discover.
        passes : int
            Number of passes through the corpus during training.
            
        Returns
        -------
        tuple: (models.LdaModel, list of str)
            The trained model and a list of human-readable topics.
        """
        if self.corpus is None or self.dictionary is None:
            raise ValueError("Corpus and dictionary must be prepared before running LDA. Call prepare_corpus() first.")
            
        lda_model = models.LdaModel(
            self.corpus,
            num_topics=num_topics,
            id2word=self.dictionary,
            passes=passes,
            random_state=42 # for reproducibility
        )
        
        # Format the topics for printing
        topics = []
        for idx, topic in lda_model.print_topics(-1):
            # Example format: 'Topic 0: 0.04..."
            topics.append(f"Topic {idx}: {topic}")
            
        return lda_model, topics
    
    def calculate_sentiment(self) -> pd.DataFrame:
        """
        Calculates sentiment scores (Compound, Pos, Neu, Neg) for all texts
        in the corpus using VADER.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'vader_compound', 'vader_pos', 'vader_neg', 'vader_neu' columns.
        """
        if self.texts is None:
            raise ValueError("Text corpus (self.texts) is empty.")
            
        # Initialize VADER
        analyzer = SentimentIntensityAnalyzer()
        
        # Apply the VADER analysis function to each text
        # This returns a dictionary of scores {'neg', 'neu', 'pos', 'compound'}
        sentiment_results = [analyzer.polarity_scores(text) for text in self.texts]
        
        # Convert list of dictionaries into a DataFrame
        sentiment_df = pd.DataFrame(sentiment_results)
        
        # Rename and reorder columns
        sentiment_df.columns = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
        sentiment_df = sentiment_df[['vader_compound', 'vader_pos', 'vader_neg', 'vader_neu']]
        
        return sentiment_df