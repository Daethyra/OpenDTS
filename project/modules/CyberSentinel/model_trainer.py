"""
ModelTrainer Module: Defines a class for building, training, and evaluating a binary classification neural network model.
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2 # Step 1: Importing L2 Regularization
import pandas as pd
import os
from dotenv import load_dotenv

class ModelTrainer:
    def __init__(self):
        load_dotenv()
        self.data_path = os.getenv('TRAINING_DATA_PATH')
        self.learning_rate = float(os.getenv('LEARNING_RATE', 0.001)) # Load Training Hyperparameters
        self.batch_size = int(os.getenv('BATCH_SIZE', 32))
        self.epochs = int(os.getenv('EPOCHS', 10))
        self.l2_reg = float(os.getenv('L2_REG', 0.01))
        if self.data_path is None:
            self.data_path = input("Enter the path to the training data file: ")
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(self.l2_reg), input_shape=(None,))) # Applying L2 Regularization
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(self.l2_reg)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def load_data(self):
        data = pd.read_csv(self.data_path)
        X = data['text']
        y = data['label']
        return X, y

    def train_model(self):
        X, y = self.load_data()
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Preprocess X_train, X_val, X_test as needed

        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.epochs, batch_size=self.batch_size) # Using Hyperparameters

        # Evaluate the model
        evaluation = self.model.evaluate(X_test, y_test)
        print("Evaluation Results:", evaluation)

        # Save the model
        self.model.save(f'CyberSentinel_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.h5')

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_model()
