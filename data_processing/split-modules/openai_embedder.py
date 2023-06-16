import openai
import os

class OpenAIEmbedder:
    def __init__(self):
        # Initialize OpenAI API
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.model_engine = "text-embeddings-ada-002"

    def generate_embedding(self, text):
        # Generate embedding for the text
        response = openai.Embedding.create(
            model=self.model_engine,
            texts=[text]
        )
        tweet_embedding = response['embeddings'][0]['embedding'] # type: ignore

        # Ensure tweet_embedding is a list of floats
        if isinstance(tweet_embedding, list) and all(isinstance(item, float) for item in tweet_embedding):
            pass
        elif isinstance(tweet_embedding, list) and all(isinstance(item, list) and len(item) == 1 and isinstance(item[0], float) for item in tweet_embedding):
            tweet_embedding = [item[0] for item in tweet_embedding]
        else:
            raise ValueError("Unexpected format for tweet_embedding")

        return tweet_embedding
