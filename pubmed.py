import requests
from bs4 import BeautifulSoup
from scraper import Scraper

class PubmedScraper(Scraper):
    def scrape(self, keywords, num_results, output_file):
        query = "+".join(keywords.split())
        url = f'https://pubmed.ncbi.nlm.nih.gov/?term={query}&size={num_results}'
        response = self.make_request(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        results = []
        for article in soup.find_all('article', class_='full-docsum'):
            title = article.find('a', class_='docsum-title').text.strip()
            authors = article.find('span', class_='authors-list').text.strip()
            snippet = article.find('div', class_='docsum_snippet').text.strip() if article.find('div', class_='docsum_snippet') else ""
            link = f"https://pubmed.ncbi.nlm.nih.gov{article.find('a', class_='docsum-title')['href']}"
            results.append({'title': title, 'authors': authors, 'snippet': snippet, 'link': link})

        self.output_results(results, output_file)
