import os
import logging
import openai
import pandas as pd
import PyPDF2
from typing import Dict, Any, Union
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    print("Please set the OpenAI API key in the environment variables.")

# Configure logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_embeddings_from_text(text: str) -> Dict[str, Any]:
    """
    Get embeddings from text using OpenAI API.
    
    :param text: The text to get embeddings from.
    :return: The embeddings in the proper format.
    """
    try:
        # Call OpenAI API to get embeddings
        response = openai.Embedding.create(model="text-embedding-ada-002", texts=[text])
        embeddings = {"vector": response["data"][0]["embeddings"]}
        return embeddings
    except Exception as e:
        logging.error(f"Error getting embeddings: {e}")
        raise
    finally:
        logging.info("Retrying getting embeddings.")

def validate_embedding(embedding: Dict[str, Any]) -> bool:
    """
    Validate the format of an embedding.
    
    :param embedding: The embedding to validate.
    :return: True if valid, False otherwise.
    """
    return isinstance(embedding, dict) and "vector" in embedding and isinstance(embedding["vector"], list)

def read_file(file_path: str) -> Union[str, None]:
    """
    Read the content of a file. Supports TXT, CSV, and PDF files.
    
    :param file_path: The path to the file.
    :return: The content of the file as a string.
    """
    _, file_extension = os.path.splitext(file_path)
    
    # Read TXT file
    if file_extension == ".txt":
        try:
            with open(file_path, "r") as file:
                return file.read()
        except Exception as e:
            logging.error(f"Error reading TXT file {file_path}: {e}")
            return None
    
    # Read CSV file
    elif file_extension == ".csv":
        try:
            # Using pandas to read CSV file
            data = pd.read_csv(file_path)
            # Convert the data to string (customize this as needed)
            return data.to_string()
        except Exception as e:
            logging.error(f"Error reading CSV file {file_path}: {e}")
            return None
    
    # Read PDF file
    elif file_extension == ".pdf":
        try:
            content = ""
            with open(file_path, "rb") as file:
                # Using PyPDF2 to read PDF file
                pdf_reader = PyPDF2.PdfFileReader(file)
                for page_num in range(pdf_reader.numPages):
                    page = pdf_reader.getPage(page_num)
                    content += page.extractText()
            return content
        except Exception as e:
            logging.error(f"Error reading PDF file {file_path}: {e}")
            return None
    
    else:
        logging.error(f"Unsupported file type: {file_extension}")
        return None

def log_info(message: str):
    """
    Log aninfo message.
    
    :param message: The message to log.
    """
    logging.info(message)

def log_error(message: str):
    """
    Log an error message.
    
    :param message: The message to log.
    """
    logging.error(message)
