import os
import asyncio
import pinecone
import utils
import glob
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

# Load configuration
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')
PINECONE_INDEX = os.getenv('PINECONE_INDEX', 'threat-embeddings')

# Initialize Pinecone
if PINECONE_API_KEY:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index = pinecone.Index(index_name=PINECONE_INDEX)
else:
    print("Please set the Pinecone API key in the environment variables.")

async def upsert_embeddings_to_pinecone(embedding: Dict[str, Any], book_id: str):
    """
    Upsert embeddings to Pinecone.
    
    :param embedding: The embedding vector.
    :param book_id: The ID of the book.
    :return: True if successful, False otherwise.
    """
    try:
        index.upsert(vectors=[(book_id, embedding)])
        utils.log_info(f"Upserted embeddings for {book_id} to Pinecone.")
        return True
    except Exception as e:
        utils.log_error(f"Error upserting embeddings: {e}")
        return False

async def process_file(file_path: str):
    """
    Process a file by reading its content, getting embeddings, and upserting to Pinecone.
    
    :param file_path: The path to the file.
    """
    content = utils.read_file(file_path)
    if content:
        embedding = utils.get_embeddings_from_text(content)
        if utils.validate_embedding(embedding):
            book_id = os.path.basename(file_path)
            success = await upsert_embeddings_to_pinecone(embedding, book_id)
            # After successfully processing a file, record it in the log file
            if success:
                utils.write_to_processed_files_log(os.path.basename(file_path))
        else:
            utils.log_error("Invalid embedding format.")
    else:
        utils.log_error(f"Failed to read content from {file_path}")

async def main():
    """
    Main asynchronous function to process files and query Pinecone index.
    """
    # Read the log file to get the list of already processed files
    processed_files = utils.read_processed_files_log()
    
    tasks = []
    for file_path in glob.glob("data/*"):
        # Check if the file has already been processed
        if os.path.basename(file_path) not in processed_files:
            task = asyncio.ensure_future(process_file(file_path))
            tasks.append(task)
    await asyncio.gather(*tasks)

def query_pinecone_index(user_query: str):
    """
    Query the Pinecone index with a user query.
    
    :param user_query: The user's query string.
    :return: The results from Pinecone.
    """
    query_embedding = utils.get_embeddings_from_text(user_query)
    results = index.query(queries=[query_embedding], top_k=5)
    return results

def chatbot_query(user_query: str):
    """
    Chatbot query function that takes a natural language query,
    retrieves similar embeddings from the Pinecone index,
    and returns the results in a human-readable format.
    
    :param user_query: The user's natural language query string.
    :return: The results in a human-readable format.
    """
    results = query_pinecone_index(user_query)
    response = "Here are the results:\n"
    for result in results:
        response += f"- {result}\n"
    return response

if __name__ == "__main__":
    asyncio.run(main())
