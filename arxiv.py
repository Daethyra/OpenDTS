import requests
from bs4 import BeautifulSoup
from base_scraper import Scraper

class ArxivScraper(Scraper):
    def __init__(self, api_key=None):
        self.api_key = api_key

    def scrape(self, keywords, num_results, output_file):
        query = "+".join(keywords)
        url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={num_results}'
        response = self.make_request(url)
        soup = BeautifulSoup(response.content, 'xml')

        results = []
        for entry in soup.find_all('entry'):
            title = entry.title.text.strip()
            authors = ', '.join([author.find('name').text for author in entry.find_all('author')])
            summary = entry.summary.text.strip()
            link = entry.id.text
            results.append({'title': title, 'authors': authors, 'summary': summary, 'link': link})

        self.output_results(results, output_file)

    def output_results(self, results, output_file):
        with open(output_file, 'w') as f:
            for result in results:
                f.write(f"Title: {result['title']}\n")
                f.write(f"Authors: {result['authors']}\n")
                f.write(f"Summary: {result['summary']}\n")
                f.write(f"Link: {result['link']}\n")
                f.write("\n" + "-" * 80 + "\n")
