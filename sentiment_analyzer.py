# sentiment_analyzer.py

import logging
import traceback
import openai
from dotenv import load_dotenv
import db_connector
from multiprocessing import Pool, cpu_count
import twtapiConn
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_tweet(tweet):
    try:
        # Use OpenAI API to analyze the tweet's sentiment
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt="Please provide a sentiment analysis for the following tweet: " + tweet.full_text,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        sentiment = response["choices"][0]["text"]

        # Use OpenAI API to moderate the tweet's content
        response = openai.Completion.create(
            engine="content-moderation-001",
            prompt=tweet.full_text,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        moderated_tweet = response["choices"][0]["text"]

        # Store the sentiment analysis and moderation results
        result = {"username": tweet.user.screen_name, "text": tweet.full_text, "sentiment": sentiment, "moderated_text": moderated_tweet}

        logging.info(f"Sentiment for tweet '{tweet.full_text}' analyzed successfully.")
        return result

    except Exception as e:
        # Log the error message and stack trace in case of any exceptions
        logging.error(f"An error occurred while analyzing tweet sentiment: {e}")
        logging.error(traceback.format_exc())

def analyze_tweet_batch(tweets):
    p = Pool(cpu_count())
    results = p.map(analyze_tweet, tweets)
    p.close()
    p.join()
    return [r for r in results if r]

def scrape_tweets(api, keywords, batch_size=50):
    try:
        # Initialize a list to store the tweets
        tweets = []

        # Scrape tweets using the keywords
        for keyword in keywords:
            # Initialize cursor
            cursor = tweepy.Cursor(api.search, q=keyword, lang="en", tweet_mode="extended", count=batch_size)

            # Scrape tweets in batches
            for i, tweet_batch in enumerate(cursor.pages()):
                # Analyze sentiment for the tweet batch in parallel
                results = analyze_tweet_batch(tweet_batch)

                # Append the results to the list of tweets
                tweets.extend(results)

                logging.info(f"Scraped batch {i+1} of tweets for keyword '{keyword}' successfully.")

        return tweets

    except Exception as e:
        # Log error
        logging.error(traceback.format_exc())
        raise e
