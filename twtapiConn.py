# twtapiConn.py

import tweepy
import logging
import traceback
import os
from dotenv import load_dotenv

load_dotenv()

def authenticate_twitter_api():
    try:
        # Authenticate to Twitter
        auth = tweepy.OAuthHandler(os.getenv("TWITTER_API_KEY"), os.getenv("TWITTER_API_SECRET_KEY"))
        auth.set_access_token(os.getenv("TWITTER_ACCESS_TOKEN"), os.getenv("TWITTER_ACCESS_TOKEN_SECRET"))
        # Create API object
        api = tweepy.API(auth)
        return api
    except Exception as e:
        # Log the error message and stack trace in case of any exceptions
        logging.error(f"An error occurred while authenticating to the Twitter API: {e}")
        logging.error(traceback.format_exc())
        raise e
