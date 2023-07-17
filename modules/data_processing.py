import pandas as pd
import logging
from tqdm.auto import tqdm
from uuid import uuid4
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from config import OPENAI_API_KEY

"""Adapt the async functionality from `abkq.py` in 'ABR/'"""

# Constants
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)  # type: ignore
index_name = 'threat-data'

# Pinecone index setup
if index_name not in pinecone.list_indexes():
    # Create a new index
    pinecone.create_index(name=index_name, metric='cosine', dimension=1536)
index = pinecone.Index(index_name)


def load_data(file_path):
    """
    Load data from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pandas.DataFrame or None: DataFrame if the file is loaded successfully, None otherwise.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error loading data from file {file_path}: {e}")
        return None


def generate_embeddings(texts):
    """
    Generate embeddings for a list of texts.
    Args:
        texts (list of str): List of texts.
    Returns:
        list of list of float: List of embeddings.
    """
    return embed.embed_documents(texts)


def send_embeddings_to_pinecone(data, batch_limit=100):
    """
    Process the provided data in batches, generate embeddings for each text in the data,
    and upsert the vectors to the Pinecone index.
    Args:
        data (pandas.DataFrame): DataFrame containing the data.
        batch_limit (int, optional): Size of the batches. Defaults to 100.
    Returns:
        None
    """
    texts = []
    metadatas = []
    for i, record in enumerate(tqdm(data)):
        metadata = {
            'wiki-id': str(record['id']),
            'source': record['url'],
            'title': record['title']
        }
        record_texts = record['text'].split()
        record_metadatas = [{"chunk": j, "text": text, **metadata} for j, text in enumerate(record_texts)]
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = generate_embeddings(texts)
            vectors = list(zip(ids, embeds, metadatas))
            index.upsert(vectors=vectors)
            texts = []
            metadatas = []
    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = generate_embeddings(texts)
        vectors = list(zip(ids, embeds, metadatas))
        index.upsert(vectors=vectors)
