# test_news_api.py
from news_api import NewsAPIScraper

def main():
    api_key = "your_newsapi_api_key"
    scraper = NewsAPIScraper(api_key)
    query = "donald trump"
    num_results = 10

    articles = scraper.scrape(query, num_results)
    headlines = scraper.get_headlines(articles)
    print("Headlines:")
    for headline in headlines:
        print(headline)

if __name__ == "__main__":
    main()
