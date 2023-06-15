# Import necessary libraries
import os
import logging
import time
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
model_engine = "text-embeddings-ada-002"

# Initialize Tweepy API
auth = tweepy.OAuthHandler(os.getenv('TWITTER_CONSUMER_KEY'), os.getenv('TWITTER_CONSUMER_SECRET'))
auth.set_access_token(os.getenv('TWITTER_ACCESS_TOKEN'), os.getenv('TWITTER_ACCESS_TOKEN_SECRET'))
api = tweepy.API(auth)

# Initialize Pinecone
pinecone.init(api_key=os.getenv('PINECONE_API_KEY', 'default_api_key')) 

# Define Pinecone index name
pinedex = os.getenv('PINECONE_INDEX', 'threat-embeddings')
index = pinecone.Index(index_name=pinedex)

# Define rate limit
rate_limit = 3

# Define PII patterns
pii_patterns = [
    '[0-9]{3}-[0-9]{2}-[0-9]{4}',  # SSN
    '[0-9]{3}-[0-9]{3}-[0-9]{4}',  # Phone number
    '\\S+@\\S+',  # Email
    '[A-Za-z0-9_ ]+',  # Twitter name
    '@\\w+'  # Twitter username
    # Add more patterns as needed
]

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



# Stream tweets in real-time
class MyStreamListener(tweepy.StreamListener): # type: ignore
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
            response = openai.Embedding.create(
                model=model_engine,
                texts=[tweet_text]
            )
            tweet_embedding = response['embeddings'][0]['embedding'] # type: ignore

            # Ensure tweet_embedding is a list of floats
            if isinstance(tweet_embedding, list) and all(isinstance(item, float) for item in tweet_embedding):
                pass
            elif isinstance(tweet_embedding, list) and all(isinstance(item, list) and len(item) == 1 and isinstance(item[0], float) for item in tweet_embedding):
                tweet_embedding = [item[0] for item in tweet_embedding]
            else:
                raise ValueError("Unexpected format for tweet_embedding")

            # Upsert the tweet ID, vector embedding, and original text to Pinecone index
            index.upsert(vectors=[(status.id_str, tweet_embedding, {'text': tweet_text})])

            # Query Pinecone index for similar tweets
            results = index.query([tweet_embedding], top_k=5, include_metadata=True)  # type: ignore

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
            # Sleep for 15 minutes
            time.sleep(15 * 60)  
            # Reset tweet count
            self.current_tweets = 0  

myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener) # type: ignore

# Start streaming tweets
myStream.filter(track=[rule['value'] for rule in rules])
