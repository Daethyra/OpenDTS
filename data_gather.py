import os
import requests
import pandas as pd
from newsapi import NewsApiClient
from dotenv import load_dotenv
from multiprocessing import Pool

load_dotenv()
RATE_LIMIT_SECONDS = 0.50

def fetch_news_articles(category):
    """
    Fetches news articles from the NewsAPI for a given category.
    
    Args:
        category (str): The news category to fetch articles for.
        
    Returns:
        pandas.DataFrame: A DataFrame containing the news articles.
    """
    # Initialize NewsAPI client with API key from environment variable
    api_key = os.getenv('NEWSAPI_API_KEY')
    newsapi = NewsApiClient(api_key=api_key)

    # Define query parameters
    query_params = {
        'category': category,
        'language': 'en',
        'page_size': 100,
        'page': 1
    }

    # Fetch news articles from NewsAPI
    response = newsapi.get_top_headlines(**query_params)
    articles = response['articles']
    
    # Create DataFrame from articles
    df = pd.DataFrame(articles)
    df['category'] = category
    
    return df

def fetch_news_articles_multiprocess(categories, max_processes=5):
    """
    Fetches news articles from the NewsAPI for multiple categories using multiprocessing.
    
    Args:
        categories (list): A list of news categories to fetch articles for.
        max_processes (int): The maximum number of processes to use.
        
    Returns:
        pandas.DataFrame: A DataFrame containing the news articles.
    """
    # Limit the number of processes to avoid hogging resources
    num_processes = min(max_processes, len(categories))
    
    # Create a process pool with the specified number of processes
    pool = Pool(processes=num_processes)
    
    # Fetch news articles for each category in parallel
    results = pool.map(fetch_news_articles, categories)
    
    # Combine the results into a single DataFrame
    df = pd.concat(results, ignore_index=True)
    
    # Save the DataFrame to a CSV file
    df.to_csv('news_articles.csv', index=False)
    
    return df
