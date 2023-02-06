import sentiment_analyzer
import db_connector
import logging
import datetime
import os

def main():
    # Configure logging
    logging.basicConfig(filename='sentiment_analysis.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    # Check if the log file is older than 7 days
    now = datetime.datetime.now()
    log_file = "sentiment_analysis.log"
    file_created = datetime.datetime.fromtimestamp(os.path.getctime(log_file))
    days_old = (now - file_created).days

    # If the log file is older than 7 days, archive it and create a new one
    if days_old >= 7:
        archive_log_file(log_file)
        open(log_file, "w").close()

    while True:
        print("1. Scrape tweets and update sentiment analysis results")
        print("2. View current sentiment analysis results")
        print("3. Update keyword list")
        print("4. Quit")

        choice = input("Enter your choice: ")

        if choice in ('1', '2', '3'):
            try:
                if choice == '1':
                    # Get the keywords from the database
                    keywords = db_connector.get_keywords()

                    # Scrape tweets using the keywords
                    tweets = sentiment_analyzer.scrape_tweets(keywords)

                    # Analyze the sentiment of the tweets
                    sentiment_results = sentiment_analyzer.analyze_sentiment(tweets)

                    # Update the sentiment analysis results in the database
                    db_connector.update_sentiment_results(sentiment_results)
                    logging.info("Sentiment analysis results updated successfully.")
                elif choice == '2':
                    # Get the current sentiment analysis results from the database
                    sentiment_results = db_connector.get_sentiment_results()

                    # Print the results
                    print(sentiment_results)
                    logging.info("Sentiment analysis results retrieved successfully.")
                elif choice == '3':
                    # Get the updated keyword list from the user
                    keywords = input("Enter the updated keyword list, separated by commas: ")
                    keywords = keywords.split(',')

                    # Update the keywords in the database
                    db_connector.update_keywords(keywords)
                    logging.info("Keywords updated successfully.")
            except Exception as e:
                logging.error("Error: {}".format(e), exc_info=True)
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

def archive_log_file(log_file):
    # Create the archive filename
    now = datetime.datetime.now()
    archive_filename = "{}-{}-{}.log".format(log_file, now.year, now.month)

    # Move the log file to the archive
    os.rename(log_file, archive_filename)
    logging.info("Log file archived as {}".format(archive_filename))
