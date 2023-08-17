""" Defines functions for ingesting files, lemmatizes and removing stop words, and tokenization. """

from dotenv import load_dotenv
import os
import re
import csv
import chardet
import logging
from PyPDF2 import PdfFileReader
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from typing import List
from datetime import datetime

# Downloading NLTK resources if not already present
nltk.download('wordnet')
nltk.download('stopwords')

# Point to the location of the .env file relative to the script's location
env_path = os.path.join(os.path.dirname(__file__), '../../../.env')

# Load the .env file
load_dotenv(dotenv_path=env_path)

logging.basicConfig(filename='preprocessing_%Y%m%d_%H%M%S.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Preprocessor:
    def __init__(self):
        self.input_file_path = os.getenv('INPUT_FILE_PATH')
        self.output_file_path = os.getenv('PREPROCESSED_DATA_FILE_PATH', f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def read_file(self, file_path: str, reader_func) -> str:
        result = ''
        if file_path == '.':
            for root, _, files in os.walk(file_path):
                for file_name in files:
                    full_path = os.path.join(root, file_name)
                    result += self.read_with_detected_encoding(full_path, reader_func)
        else:
            result = self.read_with_detected_encoding(file_path, reader_func)
        return result

    def read_with_detected_encoding(self, file_path: str, reader_func) -> str:
        with open(file_path, 'rb') as file:
            rawdata = file.read()
            result = chardet.detect(rawdata)
            encoding = result['encoding']
            file.seek(0)  # Reset the file pointer to the beginning
            try:
                text = reader_func(file, encoding)
            except Exception as e:
                logging.warning(f"Failed to process {file_path} with encoding {encoding}: {e}")
                text = ''
        return text

    def read_pdf(self, file, encoding: str) -> str:
        text = ''
        pdf_reader = PdfFileReader(file)
        for page in range(pdf_reader.getNumPages()):
            text += pdf_reader.getPage(page).extractText()
        return text

    def read_txt(self, file, encoding: str) -> List[str]:
        return [line.strip() for line in file.read().decode(encoding).splitlines() if line.strip()]

    def preprocess_text_data(self, text: str) -> str:
        # Tokenization
        tokens = re.findall(r'\w+', text)

        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Stop word removal
        tokens = [token for token in tokens if token.lower() not in self.stop_words]

        tokenized_text = " ".join(tokens)
        return tokenized_text

    def load_data(self, file_path: str) -> List[str]:
        if not os.path.exists(file_path) and file_path != '.':
            raise ValueError("File path does not exist.")
        
        if file_path.lower().endswith('.pdf'):
            raw_data = self.read_file(file_path, self.read_pdf)
        elif file_path.lower().endswith('.txt'):
            raw_data = self.read_file(file_path, self.read_txt)
        elif os.path.isdir(file_path):
            raw_data = ''
            for root, _, files in os.walk(file_path):
                for file_name in files:
                    full_path = os.path.join(root, file_name)
                    if full_path.lower().endswith('.pdf'):
                        raw_data += self.read_pdf(full_path)
                    elif full_path.lower().endswith('.txt'):
                        raw_data += self.read_txt(full_path)
        else:
            raise ValueError("Unsupported file format.")

        processed_data = [self.preprocess_text_data(text) for text in raw_data]
        return processed_data

    def save_processed_data(self, processed_data: List[str], file_type: str = "csv"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.getenv('PREPROCESSED_DATA_FILE_PATH', f"processed_data_{timestamp}.{file_type}")
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['text'])
            for text in processed_data:
                writer.writerow([text])
        print(f"Processed data saved to {file_path}")

if __name__ == "__main__":
    preprocessor = Preprocessor()
    file_path = os.getenv('INPUT_FILE_PATH') or input("Enter the path to the file or '.' to process all PDF and TXT files in the current directory: ")
    processed_data = preprocessor.load_data(file_path)
    preprocessor.save_processed_data(processed_data)
