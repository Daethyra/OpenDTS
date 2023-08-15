from dotenv import load_dotenv
import os
import csv
from typing import List, Tuple

# Point to the location of the .env file relative to the script's location
env_path = os.path.join(os.path.dirname(__file__), '../../../.env')

# Load the .env file
load_dotenv(dotenv_path=env_path)

class DataLabeler:
    def __init__(self):
        default_temp_path = os.path.dirname(__file__)
        self.temp_pdf_file_path = os.getenv('TEMP_PDF_FILE_PATH', os.path.join(default_temp_path, 'temp_pdf_data.csv'))
        self.temp_txt_file_path = os.getenv('TEMP_TXT_FILE_PATH', os.path.join(default_temp_path, 'temp_txt_data.csv'))
        self.output_file_path = os.getenv('LABELED_DATA_FILE_PATH')
        self.labeled_pdf_data = self.load_temp_data(self.temp_pdf_file_path)
        self.labeled_txt_data = self.load_temp_data(self.temp_txt_file_path)

    def get_user_input(self, prompt: str) -> bool:
        while True:
            response = input(prompt).strip().lower()
            if response in ['true', 'false']:
                return response == 'true'
            print("Invalid input! Please enter 'True' or 'False'.")

    def save_temp_data(self, data: List[Tuple[str, bool]], file_path: str):
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['text', 'label'])
            for text, label in data:
                writer.writerow([text, str(label)])

    def load_temp_data(self, file_path: str) -> List[Tuple[str, bool]]:
        data = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader)  # Skip the header
                for row in reader:
                    data.append((row[0], row[1] == 'True'))
        return data

    def label_data(self, data: List[str]) -> List[Tuple[str, bool]]:
        labeled_data = []
        for text in data:
            print(f"\\nSample:{text}")
            label = self.get_user_input("Does this text indicate the intention to commit acts of hate-based violence? (True/False): ")
            labeled_data.append((text, label))
        return labeled_data

    def save_labeled_data_to_csv(self, labeled_data: List[Tuple[str, bool]]):
        file_path = self.output_file_path or input("Enter the path to save the labeled data: ")
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['text', 'label'])
            for text, label in labeled_data:
                writer.writerow([text, str(label)])

if __name__ == "__main__":
    data_labeler = DataLabeler()
    # Load the preprocessed data from the file saved by the Preprocessor
    file_path = os.getenv('PREPROCESSED_DATA_FILE_PATH') or input("Enter the path to the preprocessed data file: ")
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        data = [row[0] for row in reader]

    labeled_data = data_labeler.label_data(data)
    data_labeler.save_labeled_data_to_csv(labeled_data)
