import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', 'default_api_key')
PINECONE_INDEX = os.getenv('PINECONE_INDEX', 'threat-embeddings')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
MODEL_ENGINE = "text-embeddings-ada-002"
