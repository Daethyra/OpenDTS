import sqlite3

def create_db():
    conn = sqlite3.connect('sentiment_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS data (source TEXT, text TEXT, sentiment REAL)''')
    conn.commit()
    conn.close()

def save_data(source, text, sentiment):
    conn = sqlite3.connect('sentiment_data.db')
    c = conn.cursor()
    c.execute("INSERT INTO data (source, text, sentiment) VALUES (?, ?, ?)", (source, text, sentiment))
    conn.commit()
    conn.close()

create_db()
