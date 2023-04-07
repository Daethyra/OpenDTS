import requests
import json
from scholarly import scholarly
from multiprocessing import Pool
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from operator import itemgetter


def search_database(args):
    db, search_keywords = args
    return globals()[db["search_func"]](search_keywords)


def main():
    databases = [
        {
            "name": "Google Scholar",
            "search_func": "search_google_scholar"
        },
        {
            "name": "PubMed",
            "search_func": "search_pubmed"
        },
        {
            "name": "arXiv",
            "search_func": "search_arxiv"
        },
        {
            "name": "IEEE Xplore",
            "search_func": "search_ieee_xplore"
        }
    ]

    search_keywords = input("Enter search keywords: ")
    num_results = int(input("Enter the number of results to display: "))

    def search_google_scholar(query):
        search_query = scholarly.search_pubs(query)
        return [next(search_query).fill() for _ in range(num_results)]

    """def search_pubmed(query):
        API_KEY = 'your_pubmed_api_key_here'
        BASE_URL = 'https://api.ncbi.nlm.nih.gov/lit/abstract/v1'
        response = requests.get(f'{BASE_URL}?db=pubmed&term={query}&retmax={num_results}&api_key={API_KEY}')
        return response.json()['result']['docs']"""

    def search_arxiv(query):
        BASE_URL = 'http://export.arxiv.org/api/query'
        response = requests.get(f'{BASE_URL}?search_query=all:{query}&start=0&max_results={num_results}')
        return response.json()['entries']

    """def search_ieee_xplore(query):
        API_KEY = 'your_ieee_api_key_here'
        BASE_URL = 'https://ieeexploreapi.ieee.org/api/v1/search/articles'
        headers = {'X-IEEE-Api-Key': API_KEY}
        params = {'querytext': query, 'max_records': num_results}
        response = requests.get(BASE_URL, headers=headers, params=params)
        return response.json()['articles']"""

    with Pool(len(databases)) as pool:
        articles_lists = pool.map(search_database, [(db, search_keywords) for db in databases])

    articles = [article for articles_list in articles_lists for article in articles_list]

    articles_with_scores = [{"article": article, "score": relevance_score(article)} for article in articles]

    sorted_articles = sorted(articles_with_scores, key=itemgetter("score"), reverse=True)

    print(f"\nTop {num_results} results:")
    for i, item in enumerate(sorted_articles[:num_results], start=1):
        print(f"{i}. {item['article']['title']} - {item['article']['author_info']}")
        print(f"Abstract: {item['article']['abstract']}\n")

if __name__ == '__main__':
    main()
