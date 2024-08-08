import pytest
import os
import boto3
import sys
import pandas as pd
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from moto import mock_aws

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from training import train_model

# Mock S3 bucket
@pytest.fixture
def s3_bucket():
    with mock_aws():
        s3 = boto3.client('s3', region_name='eu-north-1')
        s3.create_bucket(Bucket='health-ins-bucket', CreateBucketConfiguration={'LocationConstraint': 'eu-north-1'})
        s3.put_object(Bucket='health-ins-bucket', Key='data/encoded-data.csv', Body='age,sex,bmi,children,smoker,region,expenses\n19,female,27.9,0,yes,southwest,16884.924\n')
        yield s3

# Ensure the output directory exists
@pytest.fixture
def create_output_dir():
    os.makedirs('data', exist_ok=True)

# AWS S3 connection and raw data download test case
def test_aws_credentials_and_data_download(s3_bucket):
    # Ensure environment variables are set
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    assert aws_access_key_id is not None, "AWS Access Key ID is not set."
    assert aws_secret_access_key is not None, "AWS Secret Access Key is not set."
    
    # Attempt to initialize boto3 client using the fixture
    try:
        s3 = s3_bucket
    except (NoCredentialsError, PartialCredentialsError) as e:
        pytest.fail(f"AWS credentials are invalid: {e}")
    
    # Verify the object can be downloaded from S3
    try:
        obj = s3.get_object(Bucket='health-ins-bucket', Key='data/encoded-data.csv')
        encoded_data = obj['Body'].read().decode('utf-8')
        assert len(encoded_data) > 0, "Downloaded data is empty."
    except s3.exceptions.NoSuchKey:
        pytest.fail("The specified key does not exist in the bucket.")
    except Exception as e:
        pytest.fail(f"An error occurred while downloading data: {e}")

# Model training test case
def test_model_training():
    pass

# Model integration/prediction test case
def test_model_predicton():
    pass

# Error handling for incorrect AWS creds, file path, or model training test case
def test_aws_errors():
    pass