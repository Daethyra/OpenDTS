import pinecone

# Connect to the 'indicationsofviolence' index
index = pinecone.Index(name="indicationsofviolence")

def analyze_sentiment(embedding, text):
    # Query the 'indicationsofviolence' Pinecone index with the embedding
    query_results = index.query(queries=[embedding], top_k=5)

    # Infer the sentiment based on the sentiments of the most similar embeddings
    similar_sentiments = [result.id for result in query_results.results[0].matches]
    predicted_sentiment = max(set(similar_sentiments), key=similar_sentiments.count)

    print(f"Tweet: {text}\nPredicted sentiment: {predicted_sentiment}")
