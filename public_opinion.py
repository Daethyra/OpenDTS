import os
import zipfile
import pandas as pd
import tempfile

# Define a function to extract sentiment analysis results from a single file
def extract_sentiment_results(df):
    # Extract the relevant columns (e.g., username, text, sentiment) from the DataFrame
    # and return the results as a list of tuples
    results = [(row['username'], row['text'], row['sentiment']) for _, row in df.iterrows()]
    return results


# Define a function to extract sentiment analysis results from a compressed folder
def extract_sentiment_results_from_zip(filename):
    # Create a temporary directory to extract the contents of the zip file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the contents of the zip file to the temporary directory
        with zipfile.ZipFile(filename, 'r') as zip:
            zip.extractall(temp_dir)

        # Iterate over the files in the temporary directory and extract sentiment analysis results from each file
        results = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    results.extend(extract_sentiment_results(file_path))

    return results


## monitor sentiment analysis results on a topic from all files in the current working directory
# Iterate over all files and compressed folders in the current working directory
results = []
for filename in os.listdir('.'):
    if filename.endswith('.csv'):
        # If the file is a CSV file, extract sentiment analysis results from it
        results.extend(extract_sentiment_results(filename))
    elif filename.endswith('.zip'):
        # If the file is a compressed folder, extract sentiment analysis results from it
        results.extend(extract_sentiment_results_from_zip(filename))

# Load the sentiment analysis results into a pandas DataFrame
results_df = pd.DataFrame(results, columns=['username', 'text', 'sentiment'])

# Filter the results to include only tweets that mention the topic of interest
topic = str(input())
topic_results_df = results_df[results_df['text'].str.contains(topic)]

# Calculate sentiment scores for the filtered results
positive_sentiment = topic_results_df[topic_results_df['sentiment'] == 'positive'].shape[0]
negative_sentiment = topic_results_df[topic_results_df['sentiment'] == 'negative'].shape[0]
neutral_sentiment = topic_results_df[topic_results_df['sentiment'] == 'neutral'].shape[0]
total_sentiment = positive_sentiment + negative_sentiment + neutral_sentiment

# Print the sentiment scores
print(f'Sentiment toward {topic}:')
print(f'Positive: {positive_sentiment / total_sentiment:.2%}')
print(f'Negative: {negative_sentiment / total_sentiment:.2%}')
print(f'Neutral: {neutral_sentiment / total_sentiment:.2%}')

