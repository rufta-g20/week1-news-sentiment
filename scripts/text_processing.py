"""
Reusable text preprocessing and topic-modeling helpers.
"""
import re
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from gensim import corpora, models

# Hardcoded Stopwords, simplified formatting:
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
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)           # remove urls
    text = re.sub(r"[^a-z0-9\s]", " ", text)       # keep alphanum
    text = re.sub(r"\s+", " ", text).strip()
    return text

def headline_length(df: pd.DataFrame, col='headline') -> pd.Series:
    """Return length (chars and tokens) of headline column."""
    s = df[col].fillna("").astype(str)
    chars = s.str.len()
    tokens = s.apply(lambda t: len(t.split()))
    return pd.DataFrame({'chars': chars, 'tokens': tokens})

def publisher_domain(publisher: str) -> str:
    """Extract domain-like part if email-like or url-like publisher is given."""
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

def prepare_corpus(texts, no_below=5, no_above=0.5, keep_n=10000):
    texts_tok = [[w for w in clean_text(t).split() if w not in STOP and len(w) > 2] for t in texts]
    dictionary = corpora.Dictionary(texts_tok)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    corpus = [dictionary.doc2bow(text) for text in texts_tok]
    return dictionary, corpus, texts_tok

def lda_topics(dictionary, corpus, texts_tok, num_topics=6, passes=5):
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes, random_state=42)
    topics = lda.print_topics(num_words=8)
    return lda, topics