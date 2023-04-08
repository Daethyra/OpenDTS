import praw
import time
from collections import defaultdict
import spacy
import en_core_web_sm
from data_processing import analyze_sentiment
from data_storage import store_data
import data_storage

# Initialize Reddit API
reddit = praw.Reddit(client_id="YOUR_CLIENT_ID",
                     client_secret="YOUR_CLIENT_SECRET",
                     user_agent="YOUR_USER_AGENT")

nlp = en_core_web_sm.load()

# Topics for classification
TOPICS = ["politics", "sports", "technology", "entertainment", "health"]

# Determine the category of the subreddit based on its top posts
def get_subreddit_category(posts):
    topic_scores = defaultdict(int)

    for post in posts:
        post_nlp = nlp(post.title + " " + post.selftext)

        for token in post_nlp:
            if token.lemma_.lower() in TOPICS:
                topic_scores[token.lemma_.lower()] += 1

    if not topic_scores:
        return "unknown"

    # Return the topic with the highest score
    return max(topic_scores, key=topic_scores.get)

def scrape_reddit_for_keywords(keywords, limit=15):
    for keyword in keywords:
        print(f"Searching for keyword '{keyword}'")
        subreddits = reddit.subreddit("all").search(keyword, time_filter="week", limit=limit)

        for subreddit in subreddits:
            print(f"Processing subreddit: {subreddit.display_name}")

            top_posts = list(subreddit.top("week", limit=15))
            category = get_subreddit_category(top_posts)

            for post in top_posts:
                sentiment = analyze_sentiment(post.title + " " + post.selftext)
                store_data("reddit", post.title, sentiment, category)

            print(f"Category: {category}")
            time.sleep(2)  # To avoid hitting the Reddit API rate limit

def get_keywords():
    db_keywords = data_storage.get_keywords_from_db()
    print("Choose keywords from the database or enter your own. To choose from the database, enter the index of the keyword.")
    print("To enter your own keywords, type 'custom' and press Enter.")
    
    for i, keyword in enumerate(db_keywords):
        print(f"{i}: {keyword}")
    
    choice = input("Enter your choice: ")
    
    if choice.lower() == 'custom':
        print("Enter the keywords to search for on Reddit. Separate each keyword with a comma.")
        keywords_str = input("Keywords: ")
        keywords = [keyword.strip() for keyword in keywords_str.split(",")]
    else:
        index = int(choice)
        keywords = [db_keywords[index]]
    
    return keywords

if __name__ == "__main__":
    keywords = get_keywords()
    scrape_reddit_for_keywords(keywords)