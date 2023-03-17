# PhD student explanation:

## This pipeline is designed to perform predictive sentiment analysis on potential intentions of violence based on scraped news headlines and tweets. The pipeline consists of several interconnected components:

1 news_twitter_scraper.py: Scrapes the top 100 news headlines from the NewsAPI and streams tweets from Twitter using Tweepy. It uses pagination for news headlines to fetch more results.

2 sentiment_analyzer.py: Preprocesses text data and performs sentiment analysis using OpenAI's GPT-3.5-turbo API.

3 db_manager.py: Manages a SQLite database to store the sentiment analysis results, flag users surpassing the user-defined threshold for violent intentions, compresses old results, and prints the results as an ASCII table.

4 main.py: Orchestrates the pipeline by initializing the database, starting the news and tweet scraping, analyzing sentiments in parallel using multi-threading, and allowing the user to print the results or stop the scraping process.

The pipeline employs real-time tweet streaming, parallelized sentiment analysis, and data retention policies to optimize processing speed, disk memory usage, and maintainability.

# 5-year-old explanation:

## This program looks at news headlines and messages from people on Twitter to find out if someone might be saying mean things or wanting to hurt others. 
It reads the words and asks a smart computer friend if the words are happy or sad. If it finds too many sad words from the same person, it tells us that person might be dangerous. We can also look at a list of all the words and whether they were happy or sad anytime we want.