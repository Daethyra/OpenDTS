""" Defines functions for ingesting files, lemmatizes and removeing stop words, and tokenization. """

import os
import re
from PyPDF2 import PdfFileReader
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from typing import List
from datetime import datetime

# Downloading NLTK resources if not already present
nltk.download('wordnet')
nltk.download('stopwords')

class Preprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def read_pdf(self, file_path: str) -> str:
        text = ''
        if file_path == '.':
            pdf_files = [f for f in os.listdir() if os.path.isfile(f) and f.lower().endswith('.pdf')]
            for pdf_file in pdf_files:
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PdfFileReader(file)
                    for page in range(pdf_reader.getNumPages()):
                        text += pdf_reader.getPage(page).extractText()
        else:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfFileReader(file)
                for page in range(pdf_reader.getNumPages()):
                    text += pdf_reader.getPage(page).extractText()
        return text

    def read_txt(self, file_path: str) -> List[str]:
        lines = []
        if file_path == '.':
            txt_files = [f for f in os.listdir() if os.path.isfile(f) and f.lower().endswith('.txt')]
            for txt_file in txt_files:
                with open(txt_file, 'r', encoding='utf-8') as file:
                    lines += file.readlines()
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        return [line.strip() for line in lines if line.strip()]

    def preprocess_text_data(self, text: str) -> str:
        # Tokenization
        tokens = re.findall(r'\w+', text)

        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Stop word removal
        tokens = [token for token in tokens if token.lower() not in self.stop_words]

        tokenized_text = " ".join(tokens)
        return tokenized_text

    def preprocess_txt_implicit_hate_comments(self, comments: List[str]) -> List[str]:
        return [re.split(r':', comment, maxsplit=2)[-1] for comment in comments]

    def load_data(self, file_path: str) -> List[str]:
        if file_path.lower().endswith('.pdf'):
            raw_data = self.read_pdf(file_path)
        elif file_path.lower().endswith('.txt'):
            raw_data = self.read_txt(file_path)
        else:
            raise ValueError("Unsupported file format.")

        processed_data = [self.preprocess_text_data(text) for text in raw_data]
        return processed_data

    def save_processed_data(self, processed_data: List[str], file_type: str = "csv"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"processed_data_{timestamp}.{file_type}"
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['text'])
            for text in processed_data:
                writer.writerow([text])
        print(f"Processed data saved to {file_path}")

if __name__ == "__main__":
    preprocessor = Preprocessor()
    file_path = input("Enter the path to the file or '.' to process all PDF and TXT files in the current directory: ")
    processed_data = preprocessor.load_data(file_path)
    preprocessor.save_processed_data(processed_data)