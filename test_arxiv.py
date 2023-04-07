from arxiv import ArxivScraper

def main():
    api_key = "your_api_key_here"
    print("Enter the keywords separated by commas:")
    keywords = [keyword.strip() for keyword in input().split(',')]
    print("Enter the number of results:")
    num_results = int(input())
    output_file = "arxiv_results.txt"

    scraper = ArxivScraper(api_key)
    scraper.scrape(keywords, num_results, output_file)

if __name__ == "__main__":
    main()
