import vaderSentiment
import tweepy
import logging
import traceback
import pandas as pd

def scrape_tweets(keywords):
    try:
        # Authenticate to Twitter
        auth = tweepy.OAuth1UserHandler(
            consumer_key,
            consumer_secret,
            access_token,
            access_token_secret
        )

        # Create API object
        api = tweepy.API(auth)

        # Initialize a list to store the tweets
        tweets = []

        # Scrape tweets using the keywords
        for keyword in keywords:
            tweets_for_keyword = api.search(q=keyword, count=100)
            for tweet in tweets_for_keyword:
                tweets.append(tweet)

        return tweets
    except Exception as e:
