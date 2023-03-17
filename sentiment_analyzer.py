import openai
import re

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\n', ' ', text)
    return text.strip()

def get_sentiment(api_key, text):
    openai.api_key = api_key
    prompt = f"Please analyze the sentiment of the following text: {text}"
    response = openai.Completion.create(engine="gpt-3.5-turbo", prompt=prompt, max_tokens=50, n=1, stop=None, temperature=0)

    sentiment = response.choices[0].text.strip().lower()
    return sentiment
