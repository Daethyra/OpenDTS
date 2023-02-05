# OpenDTS
Open Domestic Threat Scanner - Real-time sentiment analyzer and neighborhood alert system.

The program is a sentiment analysis tool for Twitter data. It uses the Tweepy library to scrape tweets that match a set of keywords and then performs sentiment analysis on the retrieved tweets using the VADER sentiment analysis algorithm. The results of the sentiment analysis are then cached to a file so that they don't have to be recalculated every time the program is run.

The keyword list and the corresponding sentiment scores are stored in a SQLite database, and the database is periodically checked during runtime to see if any changes have been made to the keyword list. If changes have been made, the program updates which tweets are scraped.

The program uses multi-processing to speed up the sentiment analysis process by analyzing multiple tweets in parallel. The program also implements more robust error handling and logging for debugging purposes.
