# model_creator.py
import os
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from preprocessor import FileProcessor

class ModelTrainer:
    def __init__(self, data_directory):
        self.file_processor = FileProcessor(data_directory)
        self.data = self.file_processor.load_from_csv("processed_data.csv")['text']
        print(self.data.shape)

    def create_model(self):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.data)
        model = KMeans(n_clusters=5, n_init=10)
        model.fit(X)
        return model

    def save_model(self, model, filename):
        joblib.dump(model, filename)

    def load_model(self, filename):
        return joblib.load(filename)

    def train_and_save(self, filename):
        model = self.create_model()
        self.save_model(model, filename)


if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_directory, "../../data")
    model_trainer = ModelTrainer(data_directory)
    model_trainer.train_and_save("violence-indicator_model")
