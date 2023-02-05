import sqlite3

def connect_db():
    """
    Connect to the SQLite database where keywords are stored
    """
    conn = sqlite3.connect("keywords.db")
    return conn

def create_table():
    """
    Create the keywords table if it doesn't already exist in the database
    """
    conn = connect_db()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS keywords (
        keyword TEXT PRIMARY KEY,
        sentiment REAL
    )""")
    conn.commit()
    conn.close()

def add_keyword(keyword, sentiment):
    """
    Add or update a keyword in the database with its sentiment score
    """
    conn = connect_db()
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO keywords (keyword, sentiment) VALUES (?, ?)", (keyword, sentiment))
    conn.commit()
    conn.close()

def get_keywords():
    """
    Retrieve all keywords from the database
    """
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT keyword, sentiment FROM keywords")
    keywords = c.fetchall()
    conn.close()
    return keywords
