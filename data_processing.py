import openai
from functools import lru_cache
from config import API_KEY_OPENAI

openai.api_key = API_KEY_OPENAI

@lru_cache(maxsize=256)
def analyze_sentiment(text):
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",
            prompt=f"Sentiment analysis of the following text: \"{text}\". Is it positive, negative, or neutral?",
            temperature=0.5,
            max_tokens=10,
            top_p=1
        )
        sentiment = response.choices[0].text.strip().lower()
        return sentiment
    except Exception as e:
        print(f"Error in analyze_sentiment: {e}")
        return "unknown"
