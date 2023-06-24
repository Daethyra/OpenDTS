import argparse
import logging
import os
import numpy as np
import PyPDF2
import tensorflow as tf

logging.basicConfig(level=logging.INFO)


def read_pdf(file_path):
    """
    Reads text from a PDF file.
    
    :param file_path: str, path to the PDF file
    :return: str, text extracted from the PDF file or None in case of error
    """
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)
            text = ' '.join([reader.getPage(i).extractText() for i in range(reader.numPages)])
    except Exception as e:
        logging.error(f"Error reading PDF file {file_path}: {e}")
        return None

    return text


def read_data(file_path):
    """
    Reads data from a file.
    
    :param file_path: str, path to the data file
    :return: ndarray or str, data loaded from the file or None in case of error
    """
    _, file_extension = os.path.splitext(file_path)
    if file_extension == '.pdf':
        return read_pdf(file_path)
    else:
        try:
            data = np.loadtxt(file_path, delimiter=',')
        except Exception as e:
            logging.error(f"Error reading data from file {file_path}: {e}")
            return None
        return data


def standardize_features(data):
    """
    Standardize features to have mean=0 and variance=1 using TensorFlow.
    
    :param data: ndarray, data for standardization
    :return: ndarray, standardized data or None in case of error
    """
    try:
        data_std = tf.keras.utils.normalize(data)
    except Exception as e:
        logging.error(f"Error standardizing features: {e}")
        return None

    return data_std


def save_data(data, file_path):
    """
    Save the processed data to a file.
    
    :param data: ndarray, data to be saved
    :param file_path: str, path to save the data
    """
    try:
        np.savetxt(file_path, data, delimiter=',')
    except Exception as e:
        logging.error(f"Error saving data to file {file_path}: {e}")


def main(input_file_path, output_file_path):
    """
    Main function to perform preprocessing.
    
    :param input_file_path: str, path to the input data file
    :param output_file_path: str, path to save the preprocessed data
    """
    # Read data
    data = read_data(input_file_path)
    if data is None:
        return

    # Standardize features
    data_std = standardize_features(data)
    if data_std is None:
        return

    # Save the processed data
    save_data(data_std, output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for clustering.")
    parser.add_argument('--input_file_path', type=str, required=True, help="Path to the input data file.")
    parser.add_argument('--output_file_path', type=str, required=True, help="Path to save the preprocessed data.")
    args = parser.parse_args()

    main(args.input_file_path, args.output_file_path)
