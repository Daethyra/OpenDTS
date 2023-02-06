import os
from dotenv import load_dotenv
import vaderSentiment
import tweepy
import logging
import traceback
import pandas as pd

load_dotenv()

def scrape_tweets(keywords):
    try:
        # Authenticate to Twitter
        auth = tweepy.OAuth1UserHandler(
            os.getenv('CONSUMER_KEY'),
            os.getenv('CONSUMER_SECRET'),
            os.getenv('ACCESS_TOKEN'),
            os.getenv('ACCESS_TOKEN_SECRET')
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
        # Log error
        logging.error(traceback.format_exc())

def scrape_tweets_interface():
    keywords_or_hashtags = input("Do you want to scrape keywords or hashtags? (Enter 'keywords' or 'hashtags'): ")

    if keywords_or_hashtags not in ['keywords', 'hashtags']:
        print("Invalid input. Exiting program.")
        return

    keywords = []
    if keywords_or_hashtags == 'keywords':
        keywords = input("Enter up to 10 keywords, separated by a comma: ").split(',')
    else:
        keywords = input("Enter up to 10 hashtags, separated by a comma: ").split(',')

    # Call the existing scrape_tweets function with the given keywords or hashtags
    tweets = scrape_tweets(keywords)
