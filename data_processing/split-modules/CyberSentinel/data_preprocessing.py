import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is downloaded
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def load_dataset(file_path, columns):
    # Load dataset
    data = pd.read_csv(file_path, usecols=columns)
    return data

def clean_text(text_data):
    # Text cleaning: lowercase, remove punctuation, remove numbers, remove stopwords, and lemmatization
    cleaned_text_data = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    for text in text_data:
        # Convert text to lowercase
        text = text.lower()
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenization
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        cleaned_text_data.append(" ".join(tokens))
    
    return cleaned_text_data

def apply_lda(text_data, n_topics=10):
    # Vectorize the text data
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(text_data)
    
    # Perform LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    # Display topics
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        print(f"Topic #{topic_idx + 1}:")
        topic_keywords = " ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]])
        topics.append(topic_keywords)
        print(topic_keywords)
    
    return lda, topics

def tokenize_and_pad(text_data, max_len):
    # Tokenize and pad text data for neural network input
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_data)
    sequences = tokenizer.texts_to_sequences(text_data)
    word_index = tokenizer.word_index
    text_data = pad_sequences(sequences, maxlen=max_len)
    return text_data, word_index

def split_data(features, labels, test_size=0.2):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Example usage:
# Specify the file path and columns
file_path = 'path_to_your_data.csv'
columns = ['text_column', 'label_column']

# Load your data
data = load_dataset(file_path, columns)

# Clean the text data
cleaned_text_data = clean_text(data['text_column'])

# Apply LDA to cleaned text data
lda_model, topics = apply_lda(cleaned_text_data, n_topics=10)

# Tokenize and pad text data
text_data,word_index = tokenize_and_pad(cleaned_text_data, max_len=100)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = split_data(text_data, data['label_column'])

# Now, X_train, X_test, y_train, and y_test are ready for training a model.
