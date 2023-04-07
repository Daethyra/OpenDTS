# chatgpt_nlp.py
import openai

class ChatGPTAnalyzer:
    def __init__(self, api_key):
        openai.api_key = api_key

    def analyze_sentiment(self, text):
        prompt = f"Analyze the sentiment of the following text: {text}. Is it positive, negative, or neutral?"
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=5,
            n=1,
            stop=None,
            temperature=0.5,
        )

        sentiment = response.choices[0].text.strip()
        return sentiment
