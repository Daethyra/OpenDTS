import requests
import tweepy

def get_news_headlines(api_key, sources):
    url = f"https://newsapi.org/v2/top-headlines?sources={','.join(sources)}&apiKey={api_key}"
    response = requests.get(url)
    headlines = [article["title"] for article in response.json()["articles"]]
    return headlines

def get_tweets(api_key, api_secret_key, access_token, access_token_secret, threshold):
    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q="*", lang="en", result_type="recent").items(threshold):
        tweets.append(tweet.text)
    return tweets
