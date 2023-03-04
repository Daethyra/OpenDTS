import logging
import traceback
import openai
from dotenv import load_dotenv
import db_connector
import os
from word_processor import generate_word_pattern_multiprocess
import re

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def analyze_tweet(tweet):
    try:
        # Use OpenAI API to analyze the tweet's sentiment
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt="Please provide a sentiment analysis for the following tweet: " + tweet,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        sentiment = response["choices"][0]["text"]

        # Use OpenAI API to moderate the tweet's content
        response = openai.Completion.create(
            engine="content-moderation-001",
            prompt=tweet,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        moderated_tweet = response["choices"][0]["text"]

        # Store the sentiment analysis and moderation results
        result = {"text": tweet, "sentiment": sentiment, "moderated_text": moderated_tweet}

        logging.info(f"Sentiment for tweet '{tweet}' analyzed successfully.")
        return result

    except Exception as e:
        # Log the error message and stack trace in case of any exceptions
        logging.error(f"An error occurred while analyzing tweet sentiment: {e}")
        logging.error(traceback.format_exc())


def analyze_tweet_batch(tweets):
    toxic_words = db_connector.get_toxic_words()
    patterns = generate_word_pattern_multiprocess(toxic_words)
    
    results = []
    for tweet in tweets:
        # Check if the tweet contains any toxic words
        contains_toxic_words = False
        for pattern in patterns:
            if re.search(pattern, tweet, re.IGNORECASE):
                contains_toxic_words = True
                break

        # If the tweet contains toxic words, skip sentiment analysis and moderation
        if contains_toxic_words:
            continue

        # Analyze sentiment for the tweet
        result = analyze_tweet(tweet)

        # Append the result to the list of results
        if result:
            results.append(result)

    return results


def analyze_tweets():
    try:
        # Get the tweets from the database
        tweets = db_connector.get_tweets()

        # Analyze the sentiment of the tweets
        sentiment_results = analyze_tweet_batch(tweets)

        # Update the sentiment analysis and moderation results in the database
        db_connector.update_sentiment_results(sentiment_results)
        logging.info("Sentiment analysis results updated successfully.")
    except Exception as e:
        # Log the error message and stack trace in case of any exceptions
        logging.error(f"An error occurred while analyzing tweet sentiment: {e}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    analyze_tweets()
