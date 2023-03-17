import re
import openai

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\@\w+|\#','', text) # Remove @ and #
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove special characters
    text = text.lower() # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words]) # Remove stopwords
    return text

def get_sentiment(api_key, text):
    openai.api_key = api_key
    prompt = f"Sentiment analysis for: {text}. Is the sentiment positive, negative, or neutral?"
    response = openai.Completion.create(engine="gpt-3.5-turbo", prompt=prompt, max_tokens=1, n=1, temperature=0)
    sentiment = response.choices[0].text.strip()
    return sentiment
