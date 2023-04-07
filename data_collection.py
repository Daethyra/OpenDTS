import tweepy
from newsapi import NewsApiClient
from config import TWITTER_API_KEY, TWITTER_API_SECRET_KEY, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET, NEWS_API_KEY

# Set up Twitter API
auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET_KEY)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Set up NewsAPI
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

def collect_twitter_data(query, count=100):
    return api.search(q=query, count=count, lang='en', tweet_mode='extended')

def collect_news_data(query, from_date, to_date, page_size=100):
    return newsapi.get_everything(q=query, from_param=from_date, to=to_date, language='en', sort_by='relevancy', page_size=page_size)
