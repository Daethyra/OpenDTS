import asyncio
import aiohttp
import logging
from uuid import uuid4
from tqdm.auto import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from config import OPENAI_API_KEY
from preprocessor import FileProcessor

# Constants
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)  # type: ignore
index_name = 'threat-data'

# Pinecone index setup
if index_name not in pinecone.list_indexes():
    # Create a new index
    pinecone.create_index(name=index_name, metric='cosine', dimension=1536)
index = pinecone.Index(index_name)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize aiohttp session
session = aiohttp.ClientSession()

async def generate_embeddings(texts):
    """
    Asynchronously generate embeddings for a list of texts.
    """
    return embed.embed_documents(texts)

async def upsert(vectors):
    """
    Asynchronously upsert vectors to Pinecone.
    """
    index.upsert(vectors=vectors)

async def process_batch(batch):
    """
    Asynchronously process a batch of data.
    Generate embeddings for the texts in the batch and upsert them to Pinecone.
    """
    texts, metadatas = batch
    ids = [str(uuid4()) for _ in range(len(texts))]
    embeds = await generate_embeddings(texts)
    vectors = list(zip(ids, embeds, metadatas))
    await upsert(vectors)

async def send_embeddings_to_pinecone_async(data_directory, batch_limit=100):
    """
    Asynchronously process the provided data in batches.
    Generate embeddings for each text in the data and upsert the vectors to Pinecone.
    """
    tasks = []
    file_processor = FileProcessor(data_directory)
    data = file_processor.process_files()
    for record in tqdm(data.itertuples()):
        texts = record.text.split()
        metadatas = [{"chunk": j, "text": text, "wiki-id": str(record.id), "source": record.url, "title": record.title} for j, text in enumerate(texts)]
        if len(texts) >= batch_limit:
            tasks.append(asyncio.ensure_future(process_batch((texts, metadatas))))
            texts = []
            metadatas = []
    if len(texts) > 0:
        tasks.append(asyncio.ensure_future(process_batch((texts, metadatas))))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    # Load data
    data_directory = '../data'
    # Run the main function
    asyncio.run(send_embeddings_to_pinecone_async(data_directory))
    # Close the aiohttp session
    session.close()
