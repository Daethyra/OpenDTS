import tweepy
from anonymizer import Anonymizer
import asyncio
import logging
import time
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define rate limit
rate_limit = 3

class MyStreamListener(tweepy.Stream):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_tweets = 0
        self.anonymizer = Anonymizer()

    async def process_tweet(self, status):
        try:
            # Get tweet text
            if hasattr(status, 'extended_tweet'):
                tweet_text = status.extended_tweet['full_text']
            else:
                tweet_text = status.text

            # Remove PII from tweet text
            tweet_text = self.anonymizer.redact_pii(tweet_text)

        except Exception as e:
            logger.error(f"Error processing tweet {status.id_str}: {e}")

    def on_status(self, status):
        if self.current_tweets < rate_limit:
            self.current_tweets += 1
            asyncio.create_task(self.process_tweet(status))
        else:
            logger.info("Rate limit reached, waiting...")
            # Sleep for 15 minutes
            time.sleep(15 * 60)  
            # Reset tweet count
            self.current_tweets = 0  

class TwitterStream:
    def __init__(self):
        # Initialize Tweepy API
        auth = tweepy.OAuthHandler(os.getenv('TWITTER_CONSUMER_KEY'), os.getenv('TWITTER_CONSUMER_SECRET'))
        auth.set_access_token(os.getenv('TWITTER_ACCESS_TOKEN'), os.getenv('TWITTER_ACCESS_TOKEN_SECRET'))
        self.api = tweepy.API(auth)

        self.myStreamListener = MyStreamListener()
        self.myStream = tweepy.Stream(auth=self.api.auth, listener=self.myStreamListener)

    def start_streaming(self, rules):
        # Start streaming tweets
        self.myStream.filter(track=[rule['value'] for rule in rules])
