import vaderSentiment
import tweepy
import logging
import traceback
import csv

def scrape_tweets(keywords):
    try:
        # Authenticate to Twitter
        auth = tweepy.OAuth1UserHandler(
            consumer_key,
            consumer_secret,
            access_token,
            access_token_secret
        )

        # Create API object
        api = tweepy.API(auth)

        # Initialize a list to store the tweets
        tweets = []

        # Scrape tweets using the keywords
        for keyword in keywords:
            tweets_for_keyword = api.search(q=keyword, count=100)
            for tweet in tweets_for_keyword:
                tweets.append(tweet)

        return tweets
    except Exception as e:
        logging.error(f"An error occurred while scraping tweets: {e}")
        logging.error(traceback.format_exc())
        raise e

def analyze_sentiment(tweets):
    try:
        # Initialize the SentimentIntensityAnalyzer
        vader = vaderSentiment.SentimentIntensityAnalyzer()

        # Initialize a list to store the sentiment results
        sentiment_results = []

        # Analyze the sentiment of each tweet
        for tweet in tweets:
            sentiment = vader.polarity_scores(tweet.text)
            if sentiment['compound'] < -0.7:
                sentiment_results.append({'text': tweet.text, 'username': tweet.user.screen_name, 'sentiment': sentiment})

        return sentiment_results
    except Exception as e:
        logging.error(f"An error occurred while analyzing sentiment: {e}")
        logging.error(traceback.format_exc())
        raise e

def save_to_csv(sentiment_results):
    try:
        with open('sentiment_analysis.csv', 'w', newline='') as csvfile:
            fieldnames = ['username', 'text', 'sentiment']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in sentiment_results:
                writer.writerow({'username': result['username'], 'text': result['text'], 'sentiment': result['sentiment']})
    except Exception as e:
        logging.error(f"An error occurred while saving to CSV: {e}")
        logging.error(traceback.format_exc())
        raise e
