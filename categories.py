from data_gather import fetch_news_articles_multiprocess

categories = [
    'general', 'technology', 'health', 'science', 'entertainment',
    ]

df = fetch_news_articles_multiprocess(categories)

print(df.head())
