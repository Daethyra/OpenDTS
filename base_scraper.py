# base_scraper.py
import requests

class Scraper:
    def __init__(self):
        pass

    def make_request(self, url):
        response = requests.get(url)
        return response
