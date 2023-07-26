import os
import logging
import json
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tqdm import tqdm
import joblib
from PyPDF2 import PdfReader

class FileProcessor:
    def __init__(self, data_directory):
        assert isinstance(data_directory, str), "data_directory must be a string"
        logging.basicConfig(filename='file_processing.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.data_directory = data_directory
        self.lemmatizer = WordNetLemmatizer()
        logging.info('FileProcessor initialized.')

    def process_files(self):
        assert os.path.exists(self.data_directory), f'The directory "{self.data_directory}" does not exist.'
        assert os.listdir(self.data_directory), f'The directory "{self.data_directory}" is empty.'
        data = []
        for dirpath, _, filenames in tqdm(os.walk(self.data_directory), desc="Processing files"):
            logging.debug(f'Entering directory: {dirpath}')
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                logging.debug(f'Processing file: {file_path}')
                if (
                    filename.endswith(('.xls.sig', '.yml', '.md'))
                    or 'license' in filename.lower()
                ):
                    logging.info(f'Ignoring file: {file_path}')
                    continue
                try:
                    text = self.process_file(file_path)
                    data.append({'text': text, 'file_path': file_path})
                    logging.info(f'Successfully processed file: {file_path}')
                except Exception as e:
                    logging.error(f'Error processing {file_path}: {e}')
                    continue
        logging.info('File processing completed.')
        return data

    def process_file(self, file_path):
        assert isinstance(file_path, str), "file_path must be a string"
        try:
            if file_path.endswith('.pdf'):
                return self.process_pdf(file_path)
            elif file_path.endswith('.csv'):
                return self.process_csv(file_path, delimiter=',')
            elif file_path.endswith('.tsv'):
                return self.process_csv(file_path, delimiter='\t')
            elif file_path.endswith('.txt'):
                return self.process_txt(file_path)
            elif file_path.endswith('.json'):
                return self.process_json(file_path)
            elif file_path.endswith('.xls'):
                return self.process_xls(file_path)
            else:
                logging.warning(f'Unsupported file format: {file_path}')
                return ''
        except Exception as e:
            logging.error(f'Error processing {file_path}: {e}')
            return ''

    def process_pdf(self, file_path):
        logging.debug(f'Reading PDF file: {file_path}')
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = '\n'.join(page.extract_text() for page in reader.pages)
        return self.clean_text(text)

    def process_csv(self, file_path, delimiter):
        logging.debug(f'Reading CSV file: {file_path}')
        df = pd.read_csv(file_path, encoding='utf-8', delimiter=delimiter, low_memory=False)
        text = ' '.join(df.astype(str).values.flatten())
        return self.clean_text(text)

    def process_txt(self, file_path):
        logging.debug(f'Reading TXT file: {file_path}')
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return self.clean_text(text)

    def process_json(self, file_path):
        logging.debug(f'Reading JSON file: {file_path}')
        with open(file_path, 'r', encoding='utf-8') as file:
            texts = []
            for line in file:
                try:
                    data = json.loads(line)
                    text = json.dumps(data)
                    texts.append(text)
                except json.JSONDecodeError as e:
                    logging.error(f'Error decoding JSON line in {file_path}: {e}')
            return self.clean_text(' '.join(texts))

    def process_xls(self, file_path):
        logging.debug(f'Reading XLS file: {file_path}')
        df = pd.read_excel(file_path)
        text = ' '.join(df.astype(str).values.flatten())
        return self.clean_text(text)

    def save_to_csv(self, data, filename):
        df = pd.DataFrame(data, columns=['text', 'file_path'])
        df.to_csv(filename, index=False)

    def load_from_csv(self, filename):
        return pd.read_csv(filename)

    def process_and_save(self, filename):
        texts = self.process_files()
        # Check for duplicates
        df = pd.DataFrame(texts)
        duplicates = df[df.duplicated()]
        if duplicates.shape[0] > 0:
            logging.warning(f"Found {duplicates.shape[0]} duplicate records.")
            df = df.drop_duplicates()
        # Check for null values
        if df.isnull().sum().sum() > 0:
            logging.warning("Found null values, filling with empty strings.")
            df = df.fillna("")
        self.save_to_csv(df, filename)

    def clean_text(self, text):
        # Lowercase the text
        text = text.lower()
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        # Remove user @ references and '#' from text
        text = re.sub(r'\@\w+|\#','', text)
        # Remove punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove stopwords
        text_tokens = word_tokenize(text)
        text = [word for word in text_tokens if word not in stopwords.words('english')]
        # Lemmatize words
        text = [self.lemmatizer.lemmatize(word) for word in text]
        # Remove words with 2 or fewer letters
        text = [word for word in text if len(word) > 2]
        return text


class UnsupervisedModelTrainer:
    def __init__(self, data_directory):
        self.file_processor = FileProcessor(data_directory)
        self.data = self.file_processor.load_from_csv("processed_data.csv")["text"]

    def train_model(self):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.data)  # Use only the 'text' column for vectorization
        model = KMeans(n_clusters=5, n_init=10)
        model.fit(X)
        return model

    def save_model(self, model, filename):
        joblib.dump(model, filename)

    def load_model(self, filename):
        return joblib.load(filename)

    def train_and_save(self, filename):
        model = self.train_model()
        self.save_model(model, filename)


if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_directory, "../../data")
    file_processor = FileProcessor(data_directory)
    file_processor.process_and_save("processed_data.csv")

    model_trainer = UnsupervisedModelTrainer(data_directory)
    model_trainer.train_and_save("unsupervised_model.joblib")
