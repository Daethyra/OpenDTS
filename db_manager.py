import sqlite3
from prettytable import PrettyTable

def init_db():
    conn = sqlite3.connect("sentiment_results.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS sentiment_results
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       text TEXT NOT NULL,
                       sentiment TEXT NOT NULL,
                       violence_intent TEXT NOT NULL)''')
    conn.commit()
    return conn

def save_sentiment_results(conn, results):
    cursor = conn.cursor()
    cursor.executemany("INSERT INTO sentiment_results (text, sentiment, violence_intent) VALUES (?, ?, ?)", results)
    conn.commit()

def print_db_results(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sentiment_results")
    rows = cursor.fetchall()
    table = PrettyTable(["ID", "Text", "Sentiment", "Violence Intent"])
    for row in rows:
        table.add_row(row)
    print("╔══════════════════════════════════════════════════╗")
    print("║                 Sentiment Results                ║")
    print("╚══════════════════════════════════════════════════╝")
    print(table)
