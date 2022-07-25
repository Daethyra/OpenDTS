#This module uses https://github.com/bisguzar/twitter-scraper to scrape tweets related to 'anonymous' and trends across Twitter.
# This is for using twitter_scraper rather than the Twitter API itself
# This may be useful if one doesn't have access to the Twitter API.
# This module DOES CLASSIFICATION with the OpenAI API

import os
import openai
from twitter_scraper import get_tweets, get_trends

openai.api_key = os.getenv("OPENAI_API_KEY")

with open("tweets.jsonl", "r+") as tweets: #open file for reading/writing ##ERROR: does not open from current working directory -- need to learn how to connect CWD to "open"
    for tweet in get_tweets('anonymous', pages=3): #scrape twitter
        tweets.write((tweet['text']))

trends = get_trends()
tweets.write("\n",trends)

tweets.write(output)
tweets.close()

response = openai.Moderation.create(
    input = tweets
)

output = response["results"][0]