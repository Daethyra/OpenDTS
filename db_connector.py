import logging
import traceback
import openai
from dotenv import load_dotenv
import db_connector
import os
import re
from word_processor import generate_word_pattern_multiprocess

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def calculate_toxicity(tweet):
    with open("toxic_words.txt") as f:
        toxic_words = [line.strip() for line in f]

    patterns = generate_word_pattern_multiprocess(toxic_words)

    matches = []
    for pattern in patterns:
        if pattern and re.search(pattern, tweet, re.IGNORECASE):
            matches.append(pattern)

    toxicity_score = len(matches) / len(patterns)
    return toxicity_score


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
    results = []
    for tweet in tweets:
        # Calculate the toxicity score for the tweet
        toxicity_score = calculate_toxicity(tweet)

        # Analyze sentiment for the tweet
        sentiment_result = analyze_tweet(tweet)

        # Append the result to the list of results
        results.append((sentiment_result['sentiment'], sentiment_result['moderated_text'], toxicity_score))

    return results


def analyze_tweets():
    try:
        # Get the tweets from the database
        tweets = db_connector.get_tweets()

        # Analyze the sentiment and toxicity of the tweets
        results = analyze_tweet_batch(tweets)

        # Update the sentiment analysis and toxicity rated results in the database
        db_connector.update_sentiment_results(results)
        logging.info("Sentiment analysis results updated successfully.")
    except Exception as e:
        # Log the error message and stack trace in case of any exceptions
        logging.error(f"An error occurred while analyzing tweet sentiment: {e}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    analyze_tweets()
