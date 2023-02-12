import logging
from twtapiConn import authenticate_twitter_api
from sentiment_analyzer import analyze_tweet
import db_connector


def main():
    # Configure logging
    logging.basicConfig(filename=db_connector.get_log_filename(), level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    # Check if the log file is older than 7 days
    db_connector.archive_old_logs()

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

                    # Connect to the Twitter API
                    api = authenticate_twitter_api().connect_to_twitter_api()

                    # Scrape tweets using the keywords
                    tweets = []
                    for keyword in keywords:
                        cursor = api.search(q=keyword, lang="en", tweet_mode="extended")
                        for tweet in cursor:
                            tweets.append(tweet)

                    # Analyze the sentiment of the tweets
                    sentiment_results = []
                    for tweet in tweets:
                        sentiment = analyze_tweet(tweet)
                        sentiment_results.append((tweet.user.screen_name, tweet.full_text, sentiment))

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
                    # Get the current keywords from the database
                    keywords = db_connector.get_keywords()

                    # Ask the user for new keywords
                    new_keywords_str = input("Enter new keywords (comma-separated): ")

                    # Add the new keywords to the database
                    new_keywords = [k.strip() for k in new_keywords_str.split(",")]
                    keywords += new_keywords
                    db_connector.update_keywords(list(set(keywords)))

                    logging.info("Keywords updated successfully.")
            except Exception as e:
                logging.error("Error: {}".format(e), exc_info=True)
                db_connector.log_exception(e)
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    main()
