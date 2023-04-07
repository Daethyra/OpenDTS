import message_queue
from data_collection import collect_twitter_data, collect_news_data
from data_processing import analyze_sentiment
from data_storage import create_db, store_data
import time

def input_query():
    source = input("Enter the source (twitter or news): ").lower()
    query = input("Enter the query: ")

    if source == "news":
        from_date = input("Enter the start date (YYYY-MM-DD): ")
        to_date = input("Enter the end date (YYYY-MM-DD): ")
        return {'source': source, 'query': query, 'from_date': from_date, 'to_date': to_date}
    else:
        return {'source': source, 'query': query}

def main():
    create_db()
    while True:
        message = input_query()
        process_message(message)

def process_message(message):
    if message['source'] == 'twitter':
        tweets = collect_twitter_data(message['query'])
        for tweet in tweets:
            sentiment = analyze_sentiment(tweet['text'])
            is_dog_whistle = 1 if sentiment == 'negative' else 0
            store_data('twitter', tweet['text'], sentiment, is_dog_whistle)
    elif message['source'] == 'news':
        news_articles = collect_news_data(message['query'], message['from_date'], message['to_date'])
        for article in news_articles:
            sentiment = analyze_sentiment(article['text'])
            is_dog_whistle = 1 if sentiment == 'negative' else 0
            store_data('news', article['text'], sentiment, is_dog_whistle)
