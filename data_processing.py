from message_queue import send_message
from sentiment_analysis import analyze_sentiment

def process_twitter_data(tweets):
    for tweet in tweets:
        sentiment = analyze_sentiment(tweet.full_text)
        message = {'source': 'twitter', 'text': tweet.full_text, 'sentiment': sentiment}
        send_message('data_processing', message)

def process_news_data(articles):
    for article in articles['articles']:
        sentiment = analyze_sentiment(article['title'])
        message = {'source': 'news', 'text': article['title'], 'sentiment': sentiment}
        send_message('data_processing', message)
