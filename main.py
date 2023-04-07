import os
from arxiv import ArxivScraper
from ieee_xplore import IeeeXploreScraper
from pubmed import PubmedScraper
from serp import SerpScraper
from config import SERP_API_KEY, PROXIES_FILE

def load_proxies(file_path):
    with open(file_path, 'r') as file:
        proxies_list = [line.strip() for line in file.readlines()]
    return proxies

def search_scholarly_articles(keywords, num_results, output_file, source='all'):
    scrapers = {
        'arxiv': ArxivScraper(PROXIES_FILE),
        'ieee': IeeeXploreScraper(PROXIES_FILE),
        'pubmed': PubmedScraper(PROXIES_FILE),
        'serp': SerpScraper(SERP_API_KEY, PROXIES_FILE)
    }
    
    if source == 'all':
        for scraper_name, scraper in scrapers.items():
            scraper.scrape(keywords, num_results, f"{output_file}_{scraper_name}.json")
    else:
        scraper = scrapers.get(source)
        if scraper:
            scraper.scrape(keywords, num_results, output_file)
        else:
            print(f"Invalid source specified: {source}. Available sources: arxiv, ieee, pubmed, serp.")

if __name__ == "__main__":
    keywords = input("Enter the keywords: ")
    num_results = int(input("Enter the number of results: "))
    output_file = input("Enter the output file name (without extension): ")
    source = input("Enter the source (arxiv, ieee, pubmed, serp, or all): ")
    
    search_scholarly_articles(keywords, num_results, output_file, source)
