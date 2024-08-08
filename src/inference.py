import pandas as pd
import joblib
import os
import boto3
import io
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import preprocess_data
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split, cross_val_score

# Fetch AWS credentials from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Initialize a session using Amazon S3
s3 = boto3.client('s3', region_name='eu-north-1', 
                  aws_access_key_id=aws_access_key_id, 
                  aws_secret_access_key=aws_secret_access_key)

bucket_name = 'health-ins-bucket'
model_key = 'models/optimal-model-rfr.pkl'
file_key = 'data/inference-data.csv'
output_dir = 'data'

def load_model(bucket_name, model_key):
    # Download model from S3
    obj = s3.get_object(Bucket=bucket_name, Key=model_key)
    model_data = obj['Body'].read()
    
    # Load the model
    model = joblib.load(io.BytesIO(model_data))
    return model

def predict_on_inference_data(model, preprocessed_data_path):
    # Load the preprocessed data
    data = pd.read_csv(preprocessed_data_path)

    # Ensure 'expenses' column is not included in the features
    if 'expenses' in data.columns:
        data = data.drop(columns=['expenses'])

    # Predict using the loaded model
    predictions = model.predict(data)
    
    return predictions

def evaluate_model(model, X, y):
    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predict using the loaded model
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    evs = explained_variance_score(y_test, pred)
    print(f'RMSE: {rmse}')
    print(f'R^2: {r2}')
    print(f'Explained Variance Score: {evs}')
    
    # Cross-validation performance metrics
    cv_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_evs = cross_val_score(model, X_train, y_train, cv=5, scoring='explained_variance')
    
    print('Cross-Validation Performance metrics')
    print(f'CV RMSE: {cv_rmse.mean()}')
    print(f'CV R^2: {cv_r2.mean()}')
    print(f'CV Explained Variance Score: {cv_evs.mean()}')

    # Plot predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, pred, label='Predicted vs Actual')
    m, b = np.polyfit(y_test, pred, 1)
    plt.plot(y_test, m * y_test + b, color='red', label='Line of Best Fit')

    plt.xlabel('Actual Premium')
    plt.ylabel('Predicted Premium')
    plt.title(f'{model.__class__.__name__}: Predicted vs Actual')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
model = load_model(bucket_name, model_key)
preprocess_inference_data = preprocess_data(bucket_name=bucket_name, file_key=file_key, output_dir=output_dir)
preprocessed_inference_data_path = os.path.join(output_dir, 'encoded-data.csv')
predictions = predict_on_inference_data(model=model, preprocessed_data_path=preprocessed_inference_data_path)

data = pd.read_csv(preprocessed_inference_data_path)
X = data.drop(columns=['expenses'])
y = data['expenses']
# evaluate_model(model, X, y)