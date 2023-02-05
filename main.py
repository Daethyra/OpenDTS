import os
import sys
import pickle
import multiprocessing
import vaderSentiment
import tweepy
import sqlite3
import time
import logging

logger = logging.getLogger(__name__)

def connect_db():
    """
    Connect to the SQLite database where keywords are stored
    """
    conn = sqlite3.connect("keywords.db")
    return conn

def create_table():
    """
    Create the keywords table if it doesn't already exist in the database
    """
    conn = connect_db()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS keywords (
        keyword TEXT PRIMARY KEY,
        sentiment REAL
    )""")
    conn.commit()
    conn.close()

def add_keyword(keyword, sentiment):
    """
    Add or update a keyword in the database with its sentiment score
    """
    conn = connect_db()
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO keywords (keyword, sentiment) VALUES (?, ?)", (keyword, sentiment))
    conn.commit()
    conn.close()

def get_keywords():
    """
    Retrieve all keywords from the database
    """
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT keyword, sentiment FROM keywords")
    keywords = c.fetchall()
    conn.close()
    return keywords

class SentimentAnalysis:
    def __init__(self, tweets):
        self.tweets = tweets
        self.results = []
        self.cache_file = "sentiment_analysis_cache.pkl"
        self.cache_expiration = 60 # cache expires after 60 seconds

    def analyze_sentiment(self, tweet):
        analyzer = vaderSentiment.SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(tweet)
        return sentiment

    def analyze_sentiments_multiprocessing(self):
        """
        Analyze the sentiment of the tweets using the vaderSentiment library
        and multiprocessing. The results are stored in the "results" instance
        variable.
        """
        with multiprocessing.Pool() as pool:
            self.results = pool.map(self.analyze_sentiment, self.tweets)
        self.save_to_cache()

    def save_to_cache(self):
        """
        Save the sentiment analysis results to a cache file. This cache file is
        stored as a pickle file and can be loaded later to avoid repeating
        sentiment analysis.
        """
        with open(self.cache_file, "wb") as f:
            pickle.dump((self.results, time.time()), f)

    def load_from_cache(self):
        """
        Load the sentiment analysis results from the cache file if the cache
        file exists and has not expired.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                data = pickle.load(f)
                if time.time() - data[1] < self.cache_expiration:
                    self.results = data[0]
                    return True
        return False


def scrape_tweets(consumer_key, consumer_secret, access_token, access_token_secret):
    """
    Scrape tweets containing keywords and analyze their sentiment
    Continuously retrieves the list of keywords via `get_keywords` function
    """
    # initialize the Twitter API or Tweepy library with your API credentials
    auth = tweepy.OAuthHandler(
