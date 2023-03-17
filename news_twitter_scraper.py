import requests
import tweepy
from newsapi import NewsApiClient

def get_news_headlines(api_key, page_size=100, max_pages=5):
    newsapi = NewsApiClient(api_key)
    headlines = []
    for page in range(1, max_pages + 1):
        top_headlines = newsapi.get_top_headlines(language='en', page_size=page_size, page=page)
        headlines += [article['title'] for article in top_headlines['articles']]
    return headlines

class TweetStreamListener(tweepy.StreamListener):
    def __init__(self, text_queue):
        super(TweetStreamListener, self).__init__()
        self.text_queue = text_queue

    def on_status(self, status):
        self.text_queue.put(status.text)

def stream_tweets(api_key, api_secret_key, access_token, access_token_secret, text_queue):
    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    listener = TweetStreamListener(text_queue)
    stream = tweepy.Stream(auth=api.auth, listener=listener)

    stream.sample(languages=['en'])
