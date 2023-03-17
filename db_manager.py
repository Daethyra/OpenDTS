import sqlite3
import itertools
from datetime import datetime

def init_db():
    conn = sqlite3.connect('sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS results
                   (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, sentiment TEXT, violence_intention TEXT)''')
    conn.commit()
    return conn

def save_sentiment_results(conn, results):
    cursor = conn.cursor()
    cursor.executemany('INSERT INTO results (text, sentiment, violence_intention) VALUES (?, ?, ?)', results)
    conn.commit()

def print_db_results(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sentiment_results;")
    result = cursor.fetchall()
    table = PrettyTable(["ID", "Text", "Sentiment", "Violent Intent", "User ID", "Flagged"])
    for row in result:
        table.add_row(row)
    print(table)

def flag_dangerous_users(usernames, tweets, violence_intentions, threshold):
    dangerous_users = []
    tweet_violence_intentions = list(zip(usernames, tweets, violence_intentions))
    
    for user, grouped_tweets in itertools.groupby(sorted(tweet_violence_intentions, key=lambda x: x[0]), key=lambda x: x[0]):
        violent_tweets = sum(1 for tweet in grouped_tweets if tweet[2] == "Violent")
        if violent_tweets >= threshold:
            dangerous_users.append(user)
    
    return dangerous_users

def compress_old_results(conn, retention_period):
    cursor = conn.cursor()
    cutoff_date = datetime.now() - retention_period
    cursor.execute('UPDATE results SET text = "Compressed" WHERE timestamp < ?', [cutoff_date])
    conn.commit()
