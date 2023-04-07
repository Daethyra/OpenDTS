import requests
from scraper import Scraper

class SerpScraper(Scraper):

    def __init__(self, api_key, proxies_list, user_agents):
        super().__init__(proxies_list, user_agents)
        self.api_key = api_key

    def make_request(self, url):
        headers = {
            'User-Agent': self.get_random_user_agent()
        }
        proxy = self.get_random_proxy()
        response = requests.get(url, headers=headers, proxies=proxy)
        return response

    def scrape(self, keywords, num_results, output_file):
        query = " ".join(keywords.split())
        url = f'https://serpapi.com/search?q={query}&num={num_results}&api_key={self.api_key}'
        response = self.make_request(url)
        data = response.json()

        results = []
        for article in data['organic_results']:
            title = article['title']
            authors = ", ".join([author['name'] for author in article['authors']])
            snippet = article['snippet']
            link = article['link']
            results.append({'title': title, 'authors': authors, 'snippet': snippet, 'link': link})

        self.output_results(results, output_file)
