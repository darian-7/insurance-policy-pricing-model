import os
import pandas as pd
import numpy as np
import io
import boto3
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt


# Fetch AWS credentials from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Initialize a session using Amazon S3
s3 = boto3.client('s3', region_name='eu-north-1', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

bucket_name = 'health-ins-bucket'
file_key = 'data/encoded-data.csv'

def train_model(bucket_name, file_key, model_output_dir):
    # Download preprocessed data from S3
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    
    # Load the encoded data
    preprocessed_data = obj['Body'].read().decode('utf-8')
    data = pd.read_csv(io.StringIO(preprocessed_data))

    # Split the data into features and target
    X = data.drop(columns=['expenses'])
    y = data['expenses']

    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestRegressor(
        n_estimators=600,
        min_samples_split=10,
        min_samples_leaf=4
    )
    model.fit(X_train, y_train)
    
    # Save the model
    model_path = os.path.join(model_output_dir, 'model.joblib')
    joblib.dump(model, model_path)
    
    print(f"Model trained and saved at {model_path}")
    
    # Predictions and evaluation
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

    print("Model successfully trained and saved")

# Example usage
train_model(bucket_name=bucket_name, file_key=file_key, model_output_dir='models')