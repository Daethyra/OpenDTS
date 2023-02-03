import tweepy

def scrape_tweets(bigotry_keywords, num_tweets=100):
    # Initialize the API client with your API key
    auth = tweepy.OAuthHandler("CONSUMER_KEY", "CONSUMER_SECRET")
    auth.set_access_token("ACCESS_TOKEN", "ACCESS_TOKEN_SECRET")
    api = tweepy.API(auth)

    # Scrape the tweets based on the bigotry keywords
    tweets = []
    for keyword in bigotry_keywords:
        for tweet in tweepy.Cursor(api.search_tweets, q=
