import os
from dotenv import load_dotenv
from news_twitter_scraper import get_news_headlines, get_tweets
from sentiment_analyzer import preprocess_text, get_sentiment
from db_manager import init_db, save_sentiment_results, print_db_results

def predict_violence_intentions(sentiments):
    # Implement your logic to predict the intention of violence based on sentiment analysis results
    # For this example, we'll assume that any negative sentiment has the potential for violence
    return ["Violent" if sentiment == "negative" else "Non-violent" for sentiment in sentiments]

load_dotenv()

NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET_KEY = os.getenv("TWITTER_API_SECRET_KEY")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
user_defined_threshold = int(os.getenv("USER_DEFINED_THRESHOLD"))

conn = init_db()
headlines = get_news_headlines(NEWSAPI_API_KEY, 100_sources)
tweets = get_tweets(TWITTER_API_KEY, TWITTER_API_SECRET_KEY, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET, user_defined_threshold)

texts = headlines + tweets
preprocessed_texts = [preprocess_text(text) for text in texts]
sentiments = [get_sentiment(OPENAI_API_KEY, text) for text in preprocessed_texts]
violence_intentions = predict_violence_intentions(sentiments)

results = list(zip(texts, sentiments, violence_intentions))
save_sentiment_results(conn, results)

print_db_results(conn)
conn.close()
