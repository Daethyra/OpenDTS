import os
import threading
from datetime import datetime, timedelta
from dotenv import load_dotenv
from queue import Queue
from news_twitter_scraper import get_news_headlines, stream_tweets
from sentiment_analyzer import preprocess_text, get_sentiment
from db_manager import (init_db, save_sentiment_results, print_db_results, 
                        flag_dangerous_users, compress_old_results)
from spinner import start_animation

def predict_violence_intentions(sentiments):
    return ["Violent" if sentiment == "negative" else "Non-violent" for sentiment in sentiments]

def analyze_text_worker(q, api_key, stop_event):
    while not stop_event.is_set():
        try:
            text_data = q.get(timeout=1)
            preprocessed_text = preprocess_text(text_data)
            sentiment_result = get_sentiment(api_key, preprocessed_text)
            save_sentiment_results(conn, text_data, sentiment_result)
            q.task_done()

            # Check if the user is flagged
            user_id = sentiment_result["user_id"]
            if user_id:
                flagged_user = flag_dangerous_users(conn, user_id, USER_DEFINED_THRESHOLD)
                if flagged_user:
                    print(f"Alert! Dangerous user: {flagged_user}")

        except Empty:
            pass

load_dotenv()

NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET_KEY = os.getenv("TWITTER_API_SECRET_KEY")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USER_DEFINED_THRESHOLD = int(os.getenv("USER_DEFINED_THRESHOLD"))

conn = init_db()

headlines = get_news_headlines(NEWSAPI_API_KEY)
text_queue = Queue()
stop_event = threading.Event()

for i in range(5):  # Number of threads
    t = threading.Thread(target=analyze_text_worker, args=(text_queue, OPENAI_API_KEY, stop_event))
    t.daemon = True
    t.start()

for headline in headlines:
    text_queue.put(headline)

stream_thread = threading.Thread(target=stream_tweets, args=(TWITTER_API_KEY, TWITTER_API_SECRET_KEY, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET, text_queue))
stream_thread.daemon = True
stream_thread.start()

spinner_thread = threading.Thread(target=start_animation)  # Start the spinner animation
spinner_thread.daemon = True
spinner_thread.start()

try:
    while True:
        command = input("Enter 'print' to display the table or 'stop' to stop the scraping: ").strip().lower()
        if command == "print":
            stop_event.set()  # Stop the spinner temporarily
            spinner_thread.join()
            print_db_results(conn)
            flagged_users = flag_dangerous_users(conn, threshold=USER_DEFINED_THRESHOLD)
            print("Alert! Dangerous users:", flagged_users)
            spinner_thread = threading.Thread(target=start_animation)  # Restart the spinner
            spinner_thread.daemon = True
            spinner_thread.start()
            stop_event.clear()
        elif command == "stop":
            stop_event.set()
            break
finally:
    text_queue.join()
    compress_old_results(conn, timedelta(days=1))
    conn.close()