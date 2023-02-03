import csv
import vaderSentiment

# Define the bigotry keyword lists
intolerance_keywords = []
prejudice_keywords = []
bigotry_keywords = []
discrimination_keywords = []
hate_keywords = []

# Combine all the bigotry keyword lists into one
bigotry_keywords = intolerance_keywords + prejudice_keywords + bigotry_keywords + discrimination_keywords + hate_keywords

# Split the document into sentences
sentences = document.split(".")

# Create a list to store the sentences containing bigotry keywords
bigotry_sentences = [sentence for sentence in sentences if any(keyword in sentence for keyword in bigotry_keywords)]

# Create a list to store the sentiment analysis results
sentiment_results = []

# Initialize the sentiment analyzer
analyzer = vaderSentiment.SentimentIntensityAnalyzer()

# Loop through each sentence in the bigotry_sentences list
for sentence in bigotry_sentences:
    # Get the sentiment analysis result
    sentiment = analyzer.polarity_scores(sentence)

    # Add the sentiment analysis result to the sentiment_results list
    sentiment_results.append({
        "sentence": sentence,
        "sentiment": sentiment["compound"],
        "confidence": sentiment["pos"] + sentiment["neg"] + sentiment["neu"]
    })

# Write the sentiment analysis results to a CSV file
with open("sentiment_analysis_results.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(["sentence", "sentiment", "confidence"])
    writer.writerows([result.values() for result in sentiment_results])
