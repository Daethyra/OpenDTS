import logging
import traceback
import pandas as pd
import psycopg2
import psycopg2.pool
import os
import dotenv
import zipfile
from datetime import datetime, timedelta
from word_processor import is_word_valid

dotenv.load_dotenv()

ARCHIVE_FOLDER = 'archive'


def get_log_filename():
    return f"{datetime.now().strftime('%Y-%m-%d')}.log"


def log_exception(e):
    logging.error(f"An error occurred: {e}")
    logging.error(traceback.format_exc())


def connect_to_db():
    try:
        # Create a connection pool
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, # minimum number of connections in the pool
            20, # maximum number of connections in the pool
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT")
        )
        # Get a connection from the pool
        connection = connection_pool.getconn()

        # Set autocommit to True for executemany to work
        connection.autocommit = True

        # Check if the archive folder exists
        if not os.path.exists(ARCHIVE_FOLDER):
            os.makedirs(ARCHIVE_FOLDER)
            first_run = True
        else:
            # Check the creation time of the archive folder
            archive_creation_time = datetime.fromtimestamp(os.path.getctime(ARCHIVE_FOLDER))
            threshold_time = datetime.now() - timedelta(days=28)
            first_run = False

            # If the archive folder was created more than 28 days ago, archive log files and any other files meeting the threshold
            if archive_creation_time < threshold_time:
                # Create a zip file of the log files and any other files meeting the threshold
                log_files = [f for f in os.listdir() if f.endswith('.log') and os.stat(f).st_ctime < threshold_time.timestamp()]
                files_to_archive = log_files # add other files to archive here
                archive_filename = f'{ARCHIVE_FOLDER}/archive_{datetime.now().strftime("%H-%M_%Y-%m-%d")}.zip'
                with zipfile.ZipFile(archive_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zip:
                    for file in files_to_archive:
                        zip.write(file)

                # Remove the original log files and any other files that were archived
                for file in files_to_archive:
                    os.remove(file)

                # Update the creation time of the archive folder
                os.utime(ARCHIVE_FOLDER, None)

        # Return the connection with the added executemany functionality and first run flag
        return connection, connection.cursor().execute if not hasattr(connection.cursor(), 'executemany') else connection.cursor().executemany, first_run
    except Exception as e:
        # Log error
        logging.error(f"An error occurred while connecting to the database: {e}")
        logging.error(traceback.format_exc())
        raise e


def archive_old_logs():
    # Check the creation time of the log file
    log_filename = get_log_filename()
    if os.path.isfile(log_filename):
        log_creation_time = datetime.fromtimestamp(os.path.getctime(log_filename))
        threshold_time = datetime.now() - timedelta(days=7)

        # If the log file was created more than 7 days ago, archive it
        if log_creation_time < threshold_time:
            # Create a zip file of the log file
            archive_filename = f'{ARCHIVE_FOLDER}/{log_filename}.zip'
            with zipfile.ZipFile(archive_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zip:
                zip.write(log_filename)

            # Remove the original log file
            os.remove(log_filename)


def get_keywords():
    try:
        # Connect to the database
        conn, execute, _ = connect_to_db()

        # Retrieve the keywords from the database
        execute('SELECT keyword FROM keywords')
        result = execute('SELECT keyword FROM keywords')
        keywords = [row[0] for row in result]

        # Filter out any invalid keywords
        valid_keywords = [keyword for keyword in keywords if is_word_valid(keyword)]

        # Close the connection
        conn.close()

        return valid_keywords
    except Exception as e:
        # Log error
        log_exception(e)
        raise e


def update_keywords(keywords):
    try:
        # Connect to the database
        conn, execute, _ = connect_to_db()

        # Delete the old keywords
        execute('DELETE FROM keywords')

        # Insert the new keywords
        execute('INSERT INTO keywords (keyword) VALUES (%s)', [(k,) for k in keywords])

        # Close the connection
        conn.close()
    except Exception as e:
        # Log error
        log_exception(e)
        raise e


def get_sentiment_results():
    try:
        # Connect to the database
        conn, execute, _ = connect_to_db()

        # Retrieve the sentiment analysis results from the database
        execute('SELECT * FROM sentiment_results')
        result = execute('SELECT * FROM sentiment_results')
        df = pd.DataFrame(result, columns=['user', 'tweet', 'sentiment'])

        # Close the connection
        conn.close()

        return df
    except Exception as e:
        # Log error
        log_exception(e)
        raise e


def update_sentiment_results(sentiment_results):
    try:
        conn, execute, first_run = connect_to_db()

        # Create a cursor to execute SQL queries
        cursor = conn.cursor()

        # Update the sentiment analysis and moderation results in the database
        for result in sentiment_results:
            # Unpack the result tuple into sentiment, moderated_text, and toxicity_score variables
            sentiment, moderated_text, toxicity_score = result

            cursor.execute("""
                INSERT INTO sentiment_analysis_results (text, sentiment, moderated_text, toxicity_score)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (text) DO UPDATE SET
                sentiment = excluded.sentiment,
                moderated_text = excluded.moderated_text,
                toxicity_score = excluded.toxicity_score
            """, (result['text'], sentiment, moderated_text, toxicity_score))

        # Commit the changes to the database
        conn.commit()

        # Close the connection and cursor
        cursor.close()
        conn.close()

    except Exception as e:
        # Log the error message and stack trace in case of any exceptions
        logging.error(f"An error occurred while updating sentiment analysis results: {e}")
        logging.error(traceback.format_exc())
        raise e
