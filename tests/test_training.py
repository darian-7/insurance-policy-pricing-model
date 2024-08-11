import pytest
import os
import boto3
import sys
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split, cross_val_score
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from training import train_model

# Define thresholds for the model metrics test case
RMSE_THRESHOLD = 5000  # Example threshold for RMSE
R2_THRESHOLD = 0.8     # Example threshold for R^2
EVS_THRESHOLD = 0.8    # Example threshold for Explained Variance Score

# Ensure the output directory exists
@pytest.fixture
def create_output_dir():
    os.makedirs('data', exist_ok=True)

# AWS S3 connection and raw data download test case
def test_aws_credentials_and_data_download(create_output_dir):
    # Ensure environment variables are set
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region_name = os.getenv('AWS_DEFAULT_REGION', 'eu-north-1')
    bucket_name = 'health-ins-bucket'
    file_key = 'data/health-insurance.csv'
    
    assert aws_access_key_id is not None, "AWS Access Key ID is not set."
    assert aws_secret_access_key is not None, "AWS Secret Access Key is not set."
    
    # Attempt to initialize boto3 client using actual AWS credentials
    try:
        s3 = boto3.client('s3', region_name=region_name, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    except (NoCredentialsError, PartialCredentialsError) as e:
        pytest.fail(f"AWS credentials are invalid: {e}")
    
    # Verify the object can be downloaded from S3
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        raw_data = obj['Body'].read().decode('utf-8')
        assert len(raw_data) > 0, "Downloaded data is empty."
    except s3.exceptions.NoSuchKey:
        pytest.fail("The specified key does not exist in the bucket.")
    except Exception as e:
        pytest.fail(f"An error occurred while downloading data: {e}")

# Model training, integration, and prediction test case
def test_model_training(create_output_dir):
    bucket_name = 'health-ins-bucket'
    file_key = 'data/encoded-data.csv'
    
    # Call train_model to test its functionality
    train_model(bucket_name=bucket_name, file_key=file_key, model_output_dir='models')
    
    # Load the processed data
    encoded_data_path = os.path.join('data', 'encoded-data.csv')
    preprocessed_data = pd.read_csv(encoded_data_path)
    
    X = preprocessed_data.drop(columns=['expenses'])
    y = preprocessed_data['expenses']
    
    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestRegressor(
        n_estimators=600,
        min_samples_split=10,
        min_samples_leaf=4
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    # Check if the predictions are of the correct shape and type
    assert len(pred) == len(y_test), "Number of predictions does not match number of samples."
    assert isinstance(pred, np.ndarray), "Predictions are not of type np.ndarray."
    
    print("Model training and prediction tests passed.")

# Model performance metrics test case
def test_model_performance(create_output_dir):
    # Load the processed data
    encoded_data_path = os.path.join('data', 'encoded-data.csv')
    preprocessed_data = pd.read_csv(encoded_data_path)
    
    X = preprocessed_data.drop(columns=['expenses'])
    y = preprocessed_data['expenses']
    
    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load the trained model
    model = RandomForestRegressor(
        n_estimators=600,
        min_samples_split=10,
        min_samples_leaf=4
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    # Calculate and assert metrics
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    evs = explained_variance_score(y_test, pred)
    
    assert rmse < RMSE_THRESHOLD, f"RMSE is too high: {rmse} (threshold: {RMSE_THRESHOLD})"
    assert r2 > R2_THRESHOLD, f"R^2 is too low: {r2} (threshold: {R2_THRESHOLD})"
    assert evs > EVS_THRESHOLD, f"Explained Variance Score is too low: {evs} (threshold: {EVS_THRESHOLD})"
    
    print("Model performance metrics test passed.")


if __name__ == '__main__':
    pytest.main()