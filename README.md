# OpenDTS
## Open Domestic Threat Scanner - Real-time sentiment analyzer and neighborhood alert system.

This is a sentiment analysis project that performs the following operations:

Connect to a PostgreSQL database and retrieve a list of keywords.
Scrape tweets using the keywords.
Analyze the sentiment of the tweets.
Store the sentiment analysis results in the database.
Provide an option to view the current sentiment analysis results.
The project consists of two files: sentiment_analyzer.py and db_connector.py. The sentiment_analyzer.py file contains functions to scrape tweets and analyze the sentiment of the tweets. The db_connector.py file contains functions to connect to the database, retrieve the keywords and sentiment analysis results, and update the sentiment analysis results. The main.py file provides a menu-driven interface for performing the different operations.

The project also uses logging to log information and error messages. The log file is created using the basicConfig function of the logging module and the file name is "sentiment_analysis.log". The log file is checked for its age, and if it's older than 7 days, it is archived and a new log file is created.
