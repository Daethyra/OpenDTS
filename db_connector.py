import logging
import traceback
import pandas as pd
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_sentiment_results():
    """
    This function retrieves sentiment analysis results from the database.
    The results are cached using `lru_cache` to optimize performance.
    """
    try:
        # Connect to the database
        conn = connect_to_db()
        cursor = conn.cursor()

        # Query the database for the sentiment results
        query = "SELECT * FROM sentiment_analysis_results"
        cursor.execute(query)
        sentiment_results = cursor.fetchall()

        # Convert the results to a Pandas DataFrame
        df = pd.DataFrame(sentiment_results, columns=['username', 'text', 'sentiment'])

        # Close the database connection
        cursor.close()
        conn.close()

        return df
    except Exception as e:
        # Log the error message and stack trace in case of any exceptions
        logging.error(f"An error occurred while fetching sentiment results: {e}")
        logging.error(traceback.format_exc())
        raise e

@lru_cache(maxsize=1000)
def get_keywords():
    """
    This function retrieves keywords from the database.
    The results are cached using `lru_cache` to optimize performance.
    """
    try:
        # Connect to the database
        conn = connect_to_db()
        cursor = conn.cursor()

        # Query the database for the keywords
        query = "SELECT keywords FROM keyword_list"
        cursor.execute(query)
        keywords = cursor.fetchall()

        # Convert the keywords to a list
        keywords_list = [k[0] for k in keywords]

        # Close the database connection
        cursor.close()
        conn.close()

        return keywords_list
    except Exception as e:
        # Log the error message and stack trace in case of any exceptions
        logging.error(f"An error occurred while fetching keywords: {e}")
        logging.error(traceback.format_exc())
        raise e
