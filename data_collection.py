import tweepy
import requests
from config import API_KEY_TWITTER, API_KEY_NEWSAPI

# Initialize the Twitter API
auth = tweepy.AppAuthHandler(API_KEY_TWITTER, API_KEY_TWITTER)
api = tweepy.API(auth)

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
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()['articles']
        return [{'text': article['title'], 'id': article['url']} for article in news_data]
    except Exception as e:
        print(f"Error in collect_news_data: {e}")
        return []
