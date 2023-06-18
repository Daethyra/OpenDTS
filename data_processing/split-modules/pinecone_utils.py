import pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY)

# Define Pinecone index
index = pinecone.Index(index_name=PINECONE_INDEX)

def upsert_to_pinecone(tweet_id, tweet_embedding, tweet_text):
    index.upsert(vectors=[(tweet_id, tweet_embedding, {'text': tweet_text})])

def query_pinecone(tweet_embedding):
    results = index.query(queries=[tweet_embedding], top_k=5, include_metadata=True)
    return results
