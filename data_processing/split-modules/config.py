import os
from dotenv import load_dotenv
import logging
import pinecone

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI Embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

if not PINECONE_API_KEY or not PINECONE_ENV:
    raise ValueError("PINECONE_API_KEY or PINECONE_ENV not found in environment variables")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
