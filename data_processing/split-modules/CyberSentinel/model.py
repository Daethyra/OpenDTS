import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer

def create_custom_model(input_length, vocab_size, embedding_dim, lstm_units, dense_units, dropout_rate):
    model = tf.keras.Sequential()
    
    # Embedding layer
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
    
    # LSTM layers
    if isinstance(lstm_units, list):
        for units in lstm_units:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True)))
    else:
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True)))
    
    # Dense layers
    if isinstance(dense_units, list):
        for units in dense_units:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.Dropout(dropout_rate))
    else:
        model.add(tf.keras.layers.Dense(dense_units, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    return model

def create_pretrained_bert_model(model_name='bert-base-uncased'):
    # Load pre-trained BERT model
    model = TFBertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer
