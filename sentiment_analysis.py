import openai
import csv

# Initialize the API client with your API key
openai.api_key = "YOUR_API_KEY"

# Define the model and the maximum number of tokens
model_engine = "text-search-ada-doc-001"
max_tokens = 1024

# Read the document
with open("document.txt", "r") as file:
    document = file.read()

# Define the bigotry keyword lists
intolerance_keywords = []
prejudice_keywords = []
bigotry_keywords = []
discrimination_keywords = []
hate_keywords = []

# Combine all the bigotry keyword lists into one
bigotry_keywords = intolerance_keywords + prejudice_keywords + bigotry_keywords + discrimination_keywords + hate_keywords

# Split the document into sentences
sentences = document.split(".")

# Create a list to store the sentences containing bigotry keywords
bigotry_sentences = []

# Loop through each sentence
for sentence in sentences:
    # Loop through each bigotry keyword
    for keyword in bigotry_keywords:
        # If the sentence contains the keyword, add it to the bigotry_sentences list
        if keyword in sentence:
            bigotry_sentences.append(sentence)
            break

# Create a list to store the sentiment analysis results
sentiment_results = []

# Loop through each sentence in the bigotry_sentences list
for sentence in bigotry_sentences:
    # Define the prompt for sentiment analysis
    prompt = f"""
analyze_sentiment('{sentence}')
"""

    # Generate a response from the API
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Get the response message
    message = response["choices"][0]["text"]

    # Check if the token cost is within limits
    if response["choices"][0]["token_cost"] >= 1024:
        print(f"Error: Maximum token limit exceeded for sentence: {sentence}")
    elif response["choices"][0]["token_cost"] < 100:
        print(f"Error: Token cost too low for sentence: {sentence}")
    else:
        # Split the message into sentiment and confidence
        sentiment, confidence = message.split(" with ")

        # Add the sentiment analysis result to the sentiment_results list
        sentiment_results.append({
            "sentence": sentence,
            "sentiment": sentiment,
            "confidence": confidence
        })

# Write the sentiment analysis results to a CSV file
with open("sentiment_analysis_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["sentence", "sentiment", "confidence"])
    writer.writeheader()
    writer.writerows(sentiment_results)
