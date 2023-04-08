import sqlite3

def create_db():
    with sqlite3.connect('sentiment_data.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS data (source TEXT, text TEXT, sentiment TEXT, is_dog_whistle INTEGER)''')
        conn.commit()

def store_data(source, text, sentiment, is_dog_whistle):
    with sqlite3.connect('sentiment_data.db') as conn:
        c = conn.cursor()
        c.execute('INSERT INTO data VALUES (?, ?, ?, ?)', (source, text, sentiment, is_dog_whistle))
        conn.commit()

def get_keywords_from_db():
    with sqlite3.connect('sentiment_data.db') as conn:
        c = conn.cursor()
        c.execute('SELECT DISTINCT keyword FROM keywords')
        keywords = [row[0] for row in c.fetchall()]
    return keywords