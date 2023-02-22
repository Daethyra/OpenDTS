import pandas as pd
from reportlab.pdfgen import canvas
from io import BytesIO
import matplotlib.pyplot as plt


def generate_pdf_report(df):
    # Create a buffer to store the PDF file
    buffer = BytesIO()

    # Create a PDF file
    p = canvas.Canvas(buffer)

    # Create a pie chart of sentiment values
    sentiment_counts = df['sentiment'].value_counts()
    labels = sentiment_counts.index.tolist()
    sizes = sentiment_counts.tolist()
    colors = ['lightcoral', 'gold', 'yellowgreen']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Sentiment Distribution')
    plt.savefig('sentiment_distribution.png')
    p.drawImage('sentiment_distribution.png', 0, 500)

    # Create a bar chart of tweet counts by user
    user_tweet_counts = df.groupby('user')['tweet'].count().reset_index()
    plt.figure()
    plt.bar(user_tweet_counts['user'], user_tweet_counts['tweet'], color='blue')
    plt.xticks(rotation=90)
    plt.title('Tweet Counts by User')
    plt.xlabel('User')
    plt.ylabel('Tweet Count')
    plt.savefig('tweet_counts.png')
    p.drawImage('tweet_counts.png', 0, 250)

    # Close the PDF file
    p.showPage()
    p.save()

    # Get the PDF file content from the buffer
    pdf_content = buffer.getvalue()
    buffer.close()

    return pdf_content
