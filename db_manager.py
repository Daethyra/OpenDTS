import sqlite3
import itertools

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
    cursor.execute('SELECT * FROM results')
    results = cursor.fetchall()
    print("|{:^4}|{:^50}|{:^10}|{:^15}|".format("ID", "Text", "Sentiment", "Violence Intention"))
    print("-" * 82)
    for row in results:
        print("|{:^4}|{:^50}|{:^10}|{:^15}|".format(row[0], row[1][:50], row[2], row[3]))

def flag_dangerous_users(usernames, tweets, violence_intentions, threshold):
    dangerous_users = []
    tweet_violence_intentions = list(zip(usernames, tweets, violence_intentions))
    
    for user, grouped_tweets in itertools.groupby(sorted(tweet_violence_intentions, key=lambda x: x[0]), key=lambda x: x[0]):
        violent_tweets = sum(1 for tweet in grouped_tweets if tweet[2] == "Violent")
        if violent_tweets >= threshold:
            dangerous_users.append(user)
    
    return dangerous_users
