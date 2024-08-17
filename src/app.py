import joblib
import numpy as np
import os

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return "Flask API is running"

# Adjust the path to the model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/optimal-model-rfr.pkl'))
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Extract features from JSON
        features = np.array([data['age'], data['bmi'], data['children'], data['region'], data['sex_male'], data['smoker_yes']])
        features = features.reshape(1, -1)  # Reshape for a single prediction
        
        # Make a prediction
        prediction = model.predict(features)
        
        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)