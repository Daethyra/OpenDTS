import requests
import tweepy
from newsapi import NewsApiClient

def get_news_headlines(api_key, num_sources):
    newsapi = NewsApiClient(api_key)
    top_headlines = newsapi.get_top_headlines(language='en', page_size=num_sources)
    headlines = [article['title'] for article in top_headlines['articles']]
    return headlines

def get_tweets(api_key, api_secret_key, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    tweets = []
    usernames = []
    for tweet in tweepy.Cursor(api.search_tweets, q="*", lang="en", result_type="recent").items():
        tweets.append(tweet.text)
        usernames.append(tweet.user.screen_name)
    return tweets, usernames
