import os
import logging
import pandas as pd
import json
from PyPDF2 import PdfReader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

"""
Turn this module into `preprocessor.py` to sanitize and clean data before tokenization.

See `data_processing_next.py`
-> That module's next iteration calls the FileProcessor class. I believe this module is missing important functionality
--> I have no interest in splitting module functionality for the sake of modularity, at this time.

"""

class FileProcessor:
    def __init__(self, data_directory):
        assert isinstance(data_directory, str), "data_directory must be a string"
        logging.basicConfig(filename='file_processing.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.data_directory = data_directory
        logging.info('FileProcessor initialized.')

    def process_files(self):
        assert os.path.exists(self.data_directory), f'The directory "{self.data_directory}" does not exist.'
        assert os.listdir(self.data_directory), f'The directory "{self.data_directory}" is empty.'
        with ThreadPoolExecutor(max_workers=10) as executor:
            for dirpath, _, filenames in tqdm(os.walk(self.data_directory), desc="Processing files"):
                logging.debug(f'Entering directory: {dirpath}')
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    logging.debug(f'Processing file: {file_path}')
                    if (filename.endswith(('.xls.sig', '.yml', '.md')) or
                            'license' in filename.lower()):
                        logging.info(f'Ignoring file: {file_path}')
                        continue
                    executor.submit(self.process_file, file_path)
        logging.info('File processing completed.')

    def process_file(self, file_path):
        assert isinstance(file_path, str), "file_path must be a string"
        try:
            if file_path.endswith('.pdf'):
                self.process_pdf(file_path)
            elif file_path.endswith('.csv'):
                self.process_csv(file_path, delimiter=',')
            elif file_path.endswith('.tsv'):
                self.process_csv(file_path, delimiter='\t')
            elif file_path.endswith('.txt'):
                self.process_txt(file_path)
            elif file_path.endswith('.json'):
                self.process_json(file_path)
            elif file_path.endswith('.xls'):
                self.process_xls(file_path)
            else:
                logging.warning(f'Unsupported file format: {file_path}')
        except Exception as e:
            logging.error(f'Error processing {file_path}: {e}')

    def process_pdf(self, file_path):
        logging.debug(f'Reading PDF file: {file_path}')
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = '\n'.join(page.extract_text() for page in reader.pages)
        text = self.convert_to_utf8(text)

    def process_csv(self, file_path, delimiter):
        logging.debug(f'Reading CSV file: {file_path}')
        df = pd.read_csv(file_path, encoding='utf-8', delimiter=delimiter, low_memory=False)
        text = ' '.join(df.astype(str).values.flatten())
        text = self.convert_to_utf8(text)

    def process_txt(self, file_path):
        logging.debug(f'Reading TXT file: {file_path}')
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        text = self.convert_to_utf8(text)

    def process_json(self, file_path):
        logging.debug(f'Reading JSON file: {file_path}')
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    text = json.dumps(data)
                    text = self.convert_to_utf8(text)
                except json.JSONDecodeError as e:
                    logging.error(f'Error decoding JSON line in {file_path}: {e}')

    def process_xls(self, file_path):
        logging.debug(f'Reading XLS file: {file_path}')
        df = pd.read_excel(file_path)
        text = ' '.join(df.astype(str).values.flatten())
        text = self.convert_to_utf8(text)

    def convert_to_utf8(self, text):
        assert isinstance(text, (str, bytes)), "text must be a string or bytes"
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        return text

if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_directory, 'data')
    file_processor = FileProcessor(data_directory)
    file_processor.process_files()
