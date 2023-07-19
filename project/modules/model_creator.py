import os
import tensorflow as tf
import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from preprocessor import FileProcessor
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

class ModelTrainer:
    def __init__(self, data_directory):
        self.file_processor = FileProcessor(data_directory)
        self.data = self.file_processor.load_from_csv("processed_data.csv")

    def create_model(self):
        model = tf.keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(self.data.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train_model(self, model):
        train_data, test_data, train_labels, test_labels = train_test_split(
            self.data.iloc[:, :-1],
            self.data.iloc[:, -1],
            test_size=0.2
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
        tensorboard = TensorBoard(log_dir='logs')

        model.fit(
            train_data,
            train_labels,
            epochs=20,
            validation_data=(test_data, test_labels),
            callbacks=[early_stopping, model_checkpoint, tensorboard]
        )

        model.evaluate(test_data, test_labels, verbose=2)

    def save_model(self, model, filename):
        model.save(filename)

    def load_model(self, filename):
        return tf.keras.models.load_model(filename)

    def train_and_save(self, filename):
        model = self.create_model()
        self.train_model(model)
        self.save_model(model, filename)

if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_directory, "../data")
    model_trainer = ModelTrainer(data_directory)
    model_trainer.train_and_save("violence-indicator_model")