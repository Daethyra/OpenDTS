import os
import logging
import time
import asyncio
from dotenv import load_dotenv
from twitter_stream import TwitterStream
from anonymizer import Anonymizer
from openai_embedder import OpenAIEmbedder
from pinecone_upserter import PineconeUpsert
import tweepy

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define rate limit
rate_limit = 3

# Define Tweepy filtered-stream rules
rules = [
    {
        "value": "(LGBTQIA+ OR transgender OR gay OR lesbian OR bisexual OR queer OR intersex OR asexual OR genderfluid OR nonbinary) -has:links lang:en -is:retweet (context:entities:(sentiment: negative OR sentiment: very_negative))", 
        "tag": "LGBTQIA+"
    },
    {
        "value": "('Donald Trump' OR 'Matt Walsh' OR 'dont tread on me' OR 'MAGA' OR 'Second Amendment' OR 'QAnon' OR 'Proud Boys' OR 'Oath Keepers') -has:links lang:en -is:retweet (context:entities:(sentiment: negative OR sentiment: very_negative))", 
        "tag": "Right-Wing Extremism"
    },
    {
        "value": "('white power' OR 'white pride' OR 'white nationalism' OR 'white supremacy' OR 'Ku Klux Klan' OR 'neo-Nazi') -has:links lang:en -is:retweet (context:entities:(sentiment: positive OR sentiment: very_positive))", 
        "tag": "Religious Extremism"
    }
]

# Create instances of the classes
twitter_stream = TwitterStream()
anonymizer = Anonymizer()
openai_embedder = OpenAIEmbedder()
pinecone_upsert = PineconeUpsert()

# Stream tweets in real-time
class MyStreamListener(tweepy.Stream):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_tweets = 0

    async def process_tweet(self, status):
        try:
            # Get tweet text
            if hasattr(status, 'extended_tweet'):
                tweet_text = status.extended_tweet['full_text']
            else:
                tweet_text = status.text

            # Remove PII from tweet text
            tweet_text = anonymizer.redact_pii(tweet_text)

            # Generate embedding for the tweet text
            tweet_embedding = openai_embedder.generate_embedding(tweet_text)

            # Upsert the tweet ID, vector embedding, and original text to Pinecone index
            pinecone_upsert.upsert_tweet(status.id_str, tweet_embedding, tweet_text)

        except Exception as e:
            logger.error(f"Error processing tweet {status.id_str}: {e}")
            raise e
        
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

twitter_stream.myStreamListener = MyStreamListener()
twitter_stream.myStream = tweepy.Stream(auth=twitter_stream.api.auth, listener=twitter_stream.myStreamListener)

# Start streaming tweets
twitter_stream.start_streaming(rules)
