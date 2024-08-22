import pytest
import os
import boto3
import sys
import pandas as pd

from io import StringIO
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from preprocessing import preprocess_data

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region_name = os.getenv('AWS_DEFAULT_REGION', 'eu-north-1')

# Initialize boto3 client
s3 = boto3.client('s3', region_name=region_name,
                  aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key)

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


# Check for data encoding and datatype test case
def test_preprocess_data_encoding_and_datatype(create_output_dir):
    bucket_name = 'health-ins-bucket'
    file_key = 'data/encoded-data.csv'
    
    # Call preprocess_data to test its functionality
    # preprocess_data(bucket_name=bucket_name, file_key=file_key, output_dir='data')

    # Download the processed data directly from S3
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        processed_data = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    except s3.exceptions.NoSuchKey:
        pytest.fail("The specified key does not exist in the bucket.")
    except Exception as e:
        pytest.fail(f"An error occurred while downloading processed data: {e}")

    # Check if the encoding for 'sex' and 'smoker' columns has been applied
    assert 'sex_male' in processed_data.columns, "sex_male column is missing in the encoded data"
    assert 'smoker_yes' in processed_data.columns, "smoker_yes column is missing in the encoded data"

    # Check the data type of the encoded columns
    assert processed_data['sex_male'].dtype == 'int64', "sex_male column is not of type int64"
    assert processed_data['smoker_yes'].dtype == 'int64', "smoker_yes column is not of type int64"

    # Check if label encoding for 'region' column has been applied
    assert processed_data['region'].dtype == 'int32' or processed_data['region'].dtype == 'int64', "region column is not of an integer type"

    # Check if all rows have been encoded correctly
    assert not processed_data[['sex_male', 'smoker_yes', 'region']].isnull().values.any(), "Encoded columns contain null values"

    print("Data encoding and datatype change tests passed.")

# Preprocessed data upload test case
def test_data_upload(create_output_dir):
    # Ensure environment variables are set
    bucket_name = 'health-ins-bucket'
    file_key = 'data/encoded-data.csv'

    assert aws_access_key_id is not None, "AWS Access Key ID is not set."
    assert aws_secret_access_key is not None, "AWS Secret Access Key is not set."

    # Attempt to initialize boto3 client using actual AWS credentials
    s3 = boto3.client('s3', region_name=region_name, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    
    # Verify the object can be uploaded to S3
    try:
        with open(file_key, 'rb') as data_file:
            s3.upload_fileobj(data_file, bucket_name, file_key)
        
        # Verify the uploaded file by downloading it again
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        uploaded_data = obj['Body'].read().decode('utf-8')
        local_data = pd.read_csv(file_key).to_csv(index=False)
        assert uploaded_data == local_data, "Uploaded data does not match local data."
    except s3.exceptions.NoSuchKey:
        pytest.fail("The specified key does not exist in the bucket.")
    except Exception as e:
        pytest.fail(f"An error occurred while uploading or verifying data: {e}")


if __name__ == '__main__':
    pytest.main()