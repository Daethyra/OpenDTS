"""
This module is responsible for setting up the environment for the application.
It loads environment variables, sets up logging, and initializes OpenAI and Pinecone services.
"""

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
assert OPENAI_API_KEY, "OPENAI_API_KEY not found in environment variables"

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
assert PINECONE_API_KEY and PINECONE_ENV, "PINECONE_API_KEY or PINECONE_ENV not found in environment variables"
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
