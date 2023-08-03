import os
import logging
import json
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from PyPDF2 import PdfReader

class FileProcessor:
    def __init__(self, data_directory):
        assert isinstance(data_directory, str), "data_directory must be a string"
        logging.basicConfig(filename='file_processing.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.data_directory = data_directory
        self.lemmatizer = WordNetLemmatizer()
        logging.info('FileProcessor initialized.')

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\\S+|www\\S+|https\\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\\@\\w+|\\#','', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text_tokens = word_tokenize(text)
        text = [word for word in text_tokens if word not in stopwords.words('english')]
        text = [self.lemmatizer.lemmatize(word) for word in text]
        text = [word for word in text if len(word) > 2]
        return ' '.join(text)

    def process_pdf(self, file_path):
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
        return self.clean_text(text)

    def process_csv(self, file_path, delimiter=','):
        df = pd.read_csv(file_path, delimiter=delimiter)
        text = ' '.join(df.apply(lambda row: ' '.join(row.astype(str)), axis=1))
        return self.clean_text(text)

    def process_tsv(self, file_path):
        return self.process_csv(file_path, delimiter='\\t')

    def process_txt(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return self.clean_text(text)

    def process_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            text = ' '.join(str(value) for value in json_data.values())
        return self.clean_text(text)

    def process_file(self, file_path):
        assert isinstance(file_path, str), "file_path must be a string"
        try:
            if file_path.endswith('.pdf'):
                return self.process_pdf(file_path)
            elif file_path.endswith('.csv'):
                return self.process_csv(file_path)
            elif file_path.endswith('.tsv'):
                return self.process_tsv(file_path)
            elif file_path.endswith('.txt'):
                return self.process_txt(file_path)
            elif file_path.endswith('.json'):
                return self.process_json(file_path)
            else:
                logging.warning(f'Unsupported file type: {file_path}')
                return None
        except Exception as e:
            logging.error(f'Error processing {file_path}: {e}')
            return None

    def process_files(self, output_csv_filename):
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
        # Saving the processed data to CSV
        self.save_to_csv(data, output_csv_filename)

    def save_to_csv(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logging.info(f'Data saved to CSV file: {filename}')

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python preprocessor.py <data_directory> <output_csv_filename>")
        sys.exit(1)
    
    data_directory = sys.argv[1]
    output_csv_filename = sys.argv[2]

    file_processor = FileProcessor(data_directory)
    file_processor.process_files(output_csv_filename)
    print(f"Preprocessing completed. Processed data saved to {output_csv_filename}.")
