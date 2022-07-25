#Uses the Twitter API to scrape based on the listed keywords and writes them to tweets.csv
#This module does NO CLASSIFICATION work

import tweepy
import configparser
#----------------------------------------------------------------
#read configuration file
config = configparser.ConfigParser()
config.read('config.ini')
#----------------------------------------------------------------
api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']
#----------------------------------------------------------------
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']
#----------------------------------------------------------------
#authentication
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
#----------------------------------------------------------------
api = tweepy.API(auth)
#----------------------------------------------------------------
#public_tweets = api.home_timeline() ##scrape public tweets ###keep line
#----------------------------------------------------------------
keywords = ['radical left', 'ANTIFA', 'antifa', 'portland', 'lgbt', 'lgbtqia+', 'lgbtq', 'lgbtqia', 'gay', 'oregon', 'california', 'washington', 'seattle', 'los angeles', 'liberal', 'liberals', 'libtards', 'libtard', 'democrat', 'democrats',  ]
limit = 1000
#----------------------------------------------------------------
tweets = tweepy.Cursor(api.search_tweets, q=keywords, count=1000, tweet_mode='extended').items(limit)
#----------------------------------------------------------------
# create dataframe
columns = ['Time', 'User', 'Tweet']
data = []
"""for tweet in public_tweets:
    data.append([tweet.created_at, tweet.user.screen_name, tweet.text])""" #for scraping public tweets ##keep line
#----------------------------------------------------------------
for tweet in tweets:
    data.append([tweet.user.screen_name, tweet.full_text]) #for scraping by search
#----------------------------------------------------------------
df = pd.DataFrame(data, columns=columns)
#----------------------------------------------------------------
print(df)
df.to_csv('tweets.csv')