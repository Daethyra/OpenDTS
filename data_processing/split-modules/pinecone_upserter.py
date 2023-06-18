import pinecone
import os

class PineconeUpsert:
    def __init__(self):
        # Initialize Pinecone
        pinecone.init(api_key=os.getenv('PINECONE_API_KEY', 'default_api_key')) 

        # Define Pinecone index name
        pinedex = os.getenv('PINECONE_INDEX', 'threat-embeddings')
        self.index = pinecone.Index(index_name=pinedex)

    def upsert_tweet(self, id_str, tweet_embedding, tweet_text):
        # Upsert the tweet ID, vector embedding, and original text to Pinecone index
        self.index.upsert(vectors=[(id_str, tweet_embedding)], metadata={'text': tweet_text})
