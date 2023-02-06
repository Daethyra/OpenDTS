# OpenDTS
## Open Domestic Threat Scanner - Real-time sentiment analyzer and neighborhood alert system.

The program is a sentiment analysis tool for Twitter data. It uses the Tweepy library to scrape tweets that match a set of keywords and then performs sentiment analysis on the retrieved tweets using the VADER sentiment analysis algorithm. The results of the sentiment analysis are then cached to a file so that they don't have to be recalculated every time the program is run.