import tweepy
import requests
import time
from config import API_KEY_TWITTER, API_KEY_NEWSAPI
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the Twitter API
auth = tweepy.AppAuthHandler(API_KEY_TWITTER, API_KEY_TWITTER)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

session = requests.Session()
retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
    backoff_factor=1
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

def collect_twitter_data(query):
    try:
        tweets = api.search_tweets(query, count=100, lang='en', tweet_mode='extended')
        return [{'text': tweet.full_text, 'id': tweet.id} for tweet in tweets]
    except Exception as e:
        print(f"Error in collect_twitter_data: {e}")
        return []

def collect_news_data(query, from_date, to_date):
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&apiKey={API_KEY_NEWSAPI}"
        response = session.get(url, timeout=10)
        response.raise_for_status()
        news_data = response.json()['articles']
        return [{'text': article['title'], 'id': article['url']} for article in news_data]
    except Exception as e:
        print(f"Error in collect_news_data: {e}")
        return []
