import numpy as np
from data_preprocessing import load_dataset, clean_text, apply_lda, tokenize_and_pad, split_data
from model import create_custom_model
from train import compile_and_train
from evaluate import evaluate_model, model_interpretability_analysis
from utils import save_model

# Configuration
config = {
    'data_file_path': 'path_to_your_data.csv',
    'columns': ['text_column', 'label_column'],
    'model_save_path': 'path_to_save_model',
    'max_sequence_length': 100,
    'embedding_dim': 128,
    'lstm_units': 64,
    'dense_units': 32,
    'dropout_rate': 0.5,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 10,
    'n_topics': 10
}

# Load and preprocess data
data = load_dataset(config['data_file_path'], config['columns'])
cleaned_text_data = clean_text(data['text_column'])

# Apply LDA to cleaned text data
lda_model, topics = apply_lda(cleaned_text_data, config['n_topics'])

# Tokenize and pad text data
text_data, word_index = tokenize_and_pad(cleaned_text_data, config['max_sequence_length'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = split_data(text_data, data['label_column'])

# Create custom model
# Create custom model
model = create_custom_model(input_length=config['max_sequence_length'],
                            vocab_size=len(word_index) + 1,
                            embedding_dim=config['embedding_dim'],
                            lstm_units=config['lstm_units'],
                            dense_units=config['dense_units'],
                            dropout_rate=config['dropout_rate'])


# Train model
history = compile_and_train(model, X_train, y_train, config=config)

# Evaluate model
evaluate_model(model, X_test, y_test)

# Model interpretability analysis
model_interpretability_analysis(model, X_test)

# Save model
save_model(model, config['model_save_path'])
