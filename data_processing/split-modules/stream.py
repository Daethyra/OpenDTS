import requests
import json
import datetime
import asyncio
from utils import create_headers, redact_pii
from embedding import generate_embedding
from pinecone_utils import upsert_to_pinecone, query_pinecone
from config import TWITTER_BEARER_TOKEN
import logging

logger = logging.getLogger(__name__)

async def process_tweet(headers, tweet: dict):
    try:
        # Redact PII from tweet text
        tweet_text = redact_pii(tweet['data']['text'])

        # Generate embedding for the tweet text
        response = generate_embedding(tweet_text)
        
        # Type hint and check
        if isinstance(response, dict) and 'embeddings' in response:
            tweet_embedding = response['embeddings'][0]['embedding']
        else:
            logger.error(f"Unexpected response format: {response}")
            return

        # Upsert the tweet ID, vector embedding, and original text to Pinecone index
        upsert_to_pinecone(tweet['data']['id'], tweet_embedding, tweet_text)

        # Query Pinecone index for similar tweets
        results = query_pinecone(tweet_embedding)

        # Check if results are in the expected format
        if 'scores' in results and 'ids' in results:
            # Log potential threats
            for idx, score in enumerate(results['scores']):
                if score > 0.959:
                    matched_id = results['ids'][idx]
                    logger.info(f"Potential threat detected in tweet {tweet['data']['id']} with similarity score {score} to tweet {matched_id}")

    except Exception as e:
        logger.error(f"Error processing tweet {tweet['data']['id']}: {e}")

def stream_to_file_and_stdout(headers):
    response = requests.get("https://api.twitter.com/2/tweets/search/stream", headers=headers, stream=True)
    if response.status_code != 200:
        raise Exception(f"Cannot get stream (HTTP {response.status_code}): {response.text}")

    timestamp = datetime.datetime.now().strftime("%m%d%H%M")
    with open(f"twitter_stream_{timestamp}.txt", "w") as file:
        for r in response.iter_lines():
            if r:
                json_response = json.loads(r)
                print(json.dumps(json_response, indent=4))
                file.write(json.dumps(json_response) + "\n")
                asyncio.run(process_tweet(headers, json_response))
