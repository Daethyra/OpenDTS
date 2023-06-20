import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

logging.basicConfig(level=logging.INFO)


def load_data(file_path):
    """
    Load the preprocessed data from a file.
    
    :param file_path: str, path to the data file
    :return: ndarray, data loaded from the file or None in case of error
    """
    try:
        data = np.loadtxt(file_path, delimiter=',')
    except Exception as e:
        logging.error(f"Error loading data from file {file_path}: {e}")
        return None

    return data


def perform_clustering(data, n_clusters=5):
    """
    Perform K-means clustering on the data using TensorFlow.
    
    :param data: ndarray, data for clustering
    :param n_clusters: int, number of clusters
    :return: tuple, cluster centers and cluster indices or None in case of error
    """
    try:
        kmeans = tf.compat.v1.estimator.experimental.KMeans(
            num_clusters=n_clusters, use_mini_batch=False)

        # Define input function
        def input_fn():
            return tf.compat.v1.train.limit_epochs(
                tf.convert_to_tensor(data, dtype=tf.float32), num_epochs=1)

        # Train the model
        num_iterations = 10
        for _ in range(num_iterations):
            kmeans.train(input_fn)
        cluster_centers = kmeans.cluster_centers()
        cluster_indices = list(kmeans.predict_cluster_index(input_fn))
    except Exception as e:
        logging.error(f"Error performing K-means clustering: {e}")
        return None

    return cluster_centers, cluster_indices


def visualize_clusters(data, cluster_centers, cluster_indices):
    """
    Visualize the clusters using a scatter plot.
    
    :param data: ndarray, data points
    :param cluster_centers: ndarray, cluster centers
    :param cluster_indices: list, cluster indices
    """
    try:
        plt.scatter(data[:, 0], data[:, 1], c=cluster_indices, cmap='rainbow')
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='black')
        plt.show()
    except Exception as e:
        logging.error(f"Error visualizing clusters: {e}")


def main(input_file_path, n_clusters):
    """
    Main function to perform clustering.
    
    :param input_file_path: str, path to the input data file
    :param n_clusters: int, number of clusters
    """
    # Load the preprocessed data
    data = load_data(input_file_path)
    if data is None:
        return

    # Perform K-means clustering
    cluster_centers, cluster_indices = perform_clustering(data, n_clusters)
    if cluster_centers is None:
        return

    # Visualize the clusters
    visualize_clusters(data, cluster_centers, cluster_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform K-means clustering.")
    parser.add_argument('--input_file_path', type=str, required=True, help="Path to the preprocessed data file.")
    parser.add_argument('--n_clusters', type=int, default=5, help="Number of clusters for K-means.")
    args = parser.parse_args()

    main(args.input_file_path, args.n_clusters)
