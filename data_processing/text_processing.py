import openai
from sentiment_analysis import analyze_sentiment

MODEL = "text-embedding-ada-002"

def process_text(text):
    # Generate an embedding for the text
    response = openai.Embedding.create(input=[text], engine=MODEL)
    embedding = response['data'][0]['embedding']
    # Pass the embedding to the next module for further processing
    analyze_sentiment(embedding, text)
