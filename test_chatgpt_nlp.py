# test_chatgpt_nlp.py
from chatgpt_nlp import ChatGPTAnalyzer

def main():
    api_key = "your_openai_api_key"
    analyzer = ChatGPTAnalyzer(api_key)
    text = "The economy is booming, and unemployment rates are at an all-time low."
    sentiment = analyzer.analyze_sentiment(text)
    print(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
