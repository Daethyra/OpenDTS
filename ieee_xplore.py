import requests
from bs4 import BeautifulSoup
from scraper import Scraper

class IeeeXploreScraper(Scraper):
    def scrape(self, keywords, num_results, output_file):
        query = "%20".join(keywords.split())
        url = f'https://ieeexplore.ieee.org/rest/search?querytext={query}&rows={num_results}'
        response = self.make_request(url)
        data = response.json()

        results = []
        for article in data['articles']:
            title = article['title']
            authors = ', '.join([author['name'] for author in article['authors']])
            snippet = article['snippet']
            link = f"https://ieeexplore.ieee.org{article['pdfLink']}"
            results.append({'title': title, 'authors': authors, 'snippet': snippet, 'link': link})

        self.output_results(results, output_file)
