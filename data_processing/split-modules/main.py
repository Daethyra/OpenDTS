import os
from data_processing import load_data, send_embeddings_to_pinecone

def main(file_path):
    """
    Main function that loads data, generates embeddings, and sends embeddings to Pinecone.
    Args:
        file_path (str): Path to the data file.
    Returns:
        None
    """
    # Load data
    data = load_data(file_path)
    if data is None:
        return
    # Send embeddings to Pinecone
    send_embeddings_to_pinecone(data)

if __name__ == "__main__":
    # Loop through all files in the 'data/' directory
    data_directory = "./data/"
    for file_name in os.listdir(data_directory):
        # Construct the full file path
        file_path = os.path.join(data_directory, file_name)
        # Call the main function with the file path
        main(file_path)
