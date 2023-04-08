import message_queue
from data_collection import collect_twitter_data, collect_news_data
from data_processing import analyze_sentiment
from data_storage import create_db, store_data
import concurrent.futures
from reddit_scraper import process_reddit_data


def process_data(data):
    sentiment = analyze_sentiment(data['text'])
    is_dog_whistle = 1 if sentiment == 'negative' else 0
    store_data(data['source'], data['text'], sentiment, is_dog_whistle)

def process_message(message):
    if message['source'] == 'twitter':
        tweets = collect_twitter_data(message['query'])
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for tweet in tweets:
                executor.submit(process_data, {'source': 'twitter', 'text': tweet['text']})
    elif message['source'] == 'news':
        news_articles = collect_news_data(message['query'], message['from_date'], message['to_date'])
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for article in news_articles:
                executor.submit(process_data, {'source': 'news', 'text': article['text']})

if __name__ == "__main__":
    create_db()
    message_queue.receive_messages('sentiment_analysis', process_message)
    process_reddit_data()
