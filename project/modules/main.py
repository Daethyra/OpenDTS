import warnings
import asyncio
from modules.preprocessor import FileProcessor
from modules.embeddings import send_embeddings_to_pinecone
from modules.retry import retry_with_backoff


async def main():
    """
    Main function that loads data, processes it, and sends embeddings to Pinecone.
    """
    # Set data directory path
    data_directory = "../data/"

    # Set batch limit for processing
    batch_limit = 100

    # Initialize the file processor
    file_processor = FileProcessor(data_directory)

    # Process the files and retrieve the processed data
    data = await file_processor.process_files()

    # Split data into batches
    batched_data = [data[i:i + batch_limit] for i in range(0, len(data), batch_limit)]

    # Send embeddings to Pinecone for each batch
    for batch in batched_data:
        await send_embeddings_to_pinecone(batch)


if __name__ == "__main__":
    # Enable all warnings
    warnings.simplefilter("always")

    # Retry with backoff
    asyncio.run(retry_with_backoff(main))
