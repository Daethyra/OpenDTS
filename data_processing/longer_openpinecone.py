# Import necessary libraries
import os
import logging
from dotenv import load_dotenv
import pinecone
import tweepy
import openai
import asyncio
import re

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Tweepy API
auth = tweepy.OAuthHandler(os.getenv('TWITTER_CONSUMER_KEY'), os.getenv('TWITTER_CONSUMER_SECRET'))
auth.set_access_token(os.getenv('TWITTER_ACCESS_TOKEN'), os.getenv('TWITTER_ACCESS_TOKEN_SECRET'))
api = tweepy.API(auth)

# Initialize Pinecone
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'))

# Define Pinecone index name
index_name = "combined-threat-index"

# Define keywords/phrases associated with threats
threat_keywords = ["keyword1", "keyword2", "..."]

# Define rate limit
rate_limit = 3

# Define PII patterns
pii_patterns = [
    '[0-9]{3}-[0-9]{2}-[0-9]{4}',  # SSN
    '[0-9]{3}-[0-9]{3}-[0-9]{4}',  # Phone number
    '\S+@\S+',  # Email
    '[A-Za-z0-9_ ]+'  # Twitter name
    '@\\w+'  # Twitter username
    # Add more patterns as needed
]

# Initialize combined Pinecone index
combined_index = pinecone.Index(index_name=index_name)

# Stream tweets in real-time
class MyStreamListener(tweepy.StreamListener):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_tweets = 0

    async def process_tweet(self, status):
        try:
            # Get tweet text
            tweet_text = status.text

            # Remove PII from tweet text
            for pattern in pii_patterns:
                tweet_text = re.sub(pattern, '[REDACTED]', tweet_text)

            # Generate embedding for the tweet text
            tweet_embedding = openai.Embedding.create(engine="text-embedding-ada-002", prompt=tweet_text)['data'][0]['embedding']

            # Upsert the tweet ID, vector embedding, and original text to combined Pinecone index
            combined_index.upsert(vectors=[(status.id_str, tweet_embedding, {'text': tweet_text})])

            # Query combined Pinecone index for similar tweets
            results = combined_index.query([tweet_embedding], top_k=5, include_metadata=True)

            # Log potential threats
            for match in results['matches']:
                if match['score'] > 0.959:
                    logger.info(f"Potential threat detected in tweet {status.id_str} with similarity score {match['score']}")

        except Exception as e:
            logger.error(f"Error processing tweet {status.id_str}: {e}")

    def on_status(self, status):
        if self.current_tweets < rate_limit:
            self.current_tweets += 1
            asyncio.create_task(self.process_tweet(status))
        else:
            logger.info("Rate limit reached, waiting...")

myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)

# Start streaming tweets
myStream.filter(track=threat_keywords)
