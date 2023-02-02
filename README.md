# **OpenDTS**
## *Open Domestic Threat Scanner - Real-time sentiment analyzer and neighborhood alert system.*


The initial alpha release had multiple modules that allowed for Twitter API scraping, sentiment analysis, and worked with JSON files.
Now most of the program's functionality is no longer available, but is being made user friendly. I am laying out the tracks before connecting the power, so to speak.

Eventually, this program will have more telemetric functionality across the web, rather than just Twitter, but I figure it's a great place to start considering... well... how it is these days -- not that it was ever much better. ＼（〇_ｏ）／


### So what *does* it do then?

Currently the program works solely with the OpenAI embedding API endpoint to analyze the sentiment of ingested CSV file data. 

##### Next, I plan to
1) Create a scraping module for Twitter keywords and phrases with bigoted sentiment, save to CSV file
2) Assign vectors to level of intention for action
3) Figure out a way to automate alerts for people through email, a native phone application, or better yet, using the Oxen network (E2E, Onion routing).
