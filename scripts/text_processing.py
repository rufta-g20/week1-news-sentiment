
"""
Reusable text preprocessing and topic-modeling helpers.
"""
import re
import pandas as pd
import numpy as np
from gensim import corpora, models
# NOTE: nltk imports for downloads/data lookup have been removed to fix LookupError

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
    "wouldn't"
))


def clean_text(text: str) -> str:
    """
    Performs standard text cleaning: lowering case, removing punctuation, 
    and ensuring consistent whitespace.
    """
    text = text.lower()
    # Remove all non-word characters and replace with a single space.
    text = re.sub(r'[^\w\s]', ' ', text)
    # Replace multiple spaces with a single space and strip leading/trailing spaces.
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def headline_length(df: pd.DataFrame, col='headline') -> pd.Series:
    """
    Return length (chars and tokens) of headline column. 
    Uses simple string split for tokenization.
    """
    s = df[col].fillna("").astype(str)
    chars = s.str.len()
    # Fixed to use split() to bypass NLTK LookupError
    tokens = s.apply(lambda t: len(t.split()))
    return pd.DataFrame({'chars': chars, 'tokens': tokens})


def publisher_domain(publisher: str) -> str:
    """
    Extract domain-like part if email-like or url-like publisher is given.
    Requires 'tldextract' package.
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
    Manages the creation of a corpus and the training of an LDA topic model.

    This class encapsulates the gensim logic and uses the global STOP list.
    """
    def init(self):
        """Initializes the processor with empty corpus attributes."""
        self.dictionary = None
        self.corpus = None
        self.texts_tok = None

    def prepare_corpus(self, texts: list, no_below=5, no_above=0.5, keep_n=10000):
        """
        Tokenizes texts, creates a Gensim dictionary, and converts to a bag-of-words corpus.
Parameters
        ----------
        texts : list
            List of documents (headlines/body text) as strings.
        no_below : int
            Filter words that appear in fewer than no_below documents.
        no_above : float
            Filter words that appear in more than no_above proportion of documents.
        keep_n : int
            Maximum number of features (words) to keep in the final dictionary.
        """
        # Uses clean_text and simple split() for tokenization
        texts_tok = [[w for w in clean_text(t).split() if w not in STOP and len(w) > 2] for t in texts]
        
        dictionary = corpora.Dictionary(texts_tok)
        dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        corpus = [dictionary.doc2bow(text) for text in texts_tok]
        
        self.dictionary = dictionary
        self.corpus = corpus
        self.texts_tok = texts_tok

    def lda_topics(self, num_topics=6, passes=6):
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
            raise ValueError("Corpus and dictionary must be prepared before running LDA.")
            
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
            # Example format: 'Topic 0: 0.045*"stock" + 0.030*"price" + ...'
            topics.append(f"Topic {idx}: {topic}")
            
        return lda_model, topics