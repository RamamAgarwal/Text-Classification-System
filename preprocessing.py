"""
Text Preprocessing Module
Handles cleaning, tokenization, stopwords removal, and feature extraction
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Newer versions of NLTK also require the punkt_tab resource
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        # If this fails, the user can manually run:
        # import nltk; nltk.download('punkt_tab')
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    """Class for preprocessing text data"""
    
    def __init__(self, remove_stopwords=True, stem_words=True):
        """
        Initialize the preprocessor
        
        Args:
            remove_stopwords: Whether to remove stopwords
            stem_words: Whether to stem words
        """
        self.remove_stopwords = remove_stopwords
        self.stem_words = stem_words
        self.stemmer = PorterStemmer() if stem_words else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
    def clean_text(self, text):
        """
        Clean text by removing special characters, converting to lowercase
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords_from_tokens(self, tokens):
        """
        Remove stopwords from token list
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of tokens without stopwords
        """
        if not self.remove_stopwords:
            return tokens
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens):
        """
        Stem tokens using Porter Stemmer
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of stemmed tokens
        """
        if not self.stem_words or self.stemmer is None:
            return tokens
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline for a single text
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Remove stopwords
        tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Stem tokens
        tokens = self.stem_tokens(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_corpus(self, texts):
        """
        Preprocess a collection of texts
        
        Args:
            texts: List or Series of text strings
            
        Returns:
            List of preprocessed text strings
        """
        return [self.preprocess_text(text) for text in texts]


class FeatureExtractor:
    """Class for extracting numerical features from text"""
    
    def __init__(self, method='tfidf', max_features=5000, ngram_range=(1, 2)):
        """
        Initialize feature extractor
        
        Args:
            method: 'tfidf' or 'bow' (Bag of Words)
            max_features: Maximum number of features
            ngram_range: Range of n-grams to use
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.95
            )
        else:  # bow
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.95
            )
    
    def fit_transform(self, texts):
        """
        Fit vectorizer and transform texts to feature matrix
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Feature matrix (sparse array)
        """
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """
        Transform texts to feature matrix using fitted vectorizer
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Feature matrix (sparse array)
        """
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names from vectorizer"""
        return self.vectorizer.get_feature_names_out()
