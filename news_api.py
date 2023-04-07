# news_api.py
import requests

class NewsAPIScraper:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"

    def scrape(self, query, num_results=10):
        params = {
            "q": query,
            "apiKey": self.api_key,
            "pageSize": num_results
        }
        response = requests.get(self.base_url, params=params)

        if response.status_code == 200:
            return response.json()["articles"]
        else:
            return []

    def get_headlines(self, articles):
        headlines = [article["title"] for article in articles]
        return headlines
