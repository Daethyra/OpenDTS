import tensorflow as tf
from flask import Flask, request, jsonify
import json

def save_model(model, path):
    model.save(path)

def load_model(path):
    return tf.keras.models.load_model(path)

def make_predictions(model, data):
    return model.predict(data)

def deploy_model_as_api(model, host='0.0.0.0', port=5000):
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json(force=True)
        predictions = make_predictions(model, data['input'])
        return jsonify(predictions.tolist())

    app.run(host=host, port=port)

def export_model_for_mobile(model, path):
    # Convert the model to the TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the model to disk
    with open(path, 'wb') as f:
        f.write(tflite_model)
