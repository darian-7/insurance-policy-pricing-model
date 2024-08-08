import pytest
import os
import boto3
import sys
import pandas as pd
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from moto import mock_aws

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from preprocessing import preprocess_data

# Mock S3 bucket
@pytest.fixture
def s3_bucket():
    with mock_aws():
        s3 = boto3.client('s3', region_name='eu-north-1')
        s3.create_bucket(Bucket='health-ins-bucket', CreateBucketConfiguration={'LocationConstraint': 'eu-north-1'})
        s3.put_object(Bucket='health-ins-bucket', Key='data/health-insurance.csv', Body='age,sex,bmi,children,smoker,region,expenses\n19,female,27.9,0,yes,southwest,16884.924\n')
        yield s3

# Ensure the output directory exists
@pytest.fixture
def create_output_dir():
    os.makedirs('data', exist_ok=True)

# AWS S3 connection and raw data download test case
def test_aws_credentials_and_data_download(s3_bucket, create_output_dir):
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
        obj = s3.get_object(Bucket='health-ins-bucket', Key='data/health-insurance.csv')
        raw_data = obj['Body'].read().decode('utf-8')
        assert len(raw_data) > 0, "Downloaded data is empty."
    except s3.exceptions.NoSuchKey:
        pytest.fail("The specified key does not exist in the bucket.")
    except Exception as e:
        pytest.fail(f"An error occurred while downloading data: {e}")

def test_preprocess_data_encoding_and_datatype(s3_bucket, create_output_dir):
    # Call preprocess_data to test its functionality
    preprocess_data(bucket_name='health-ins-bucket', file_key='data/health-insurance.csv', output_dir='data')

    # Load the processed data
    processed_data = pd.read_csv('data/encoded-data.csv')

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

# Preprocessed data upload back to S3 test case
def test_upload_preprocessed_data_s3():
    pass

# Error handling for incorrect AWS creds or file path test case
def test_aws_errors():
    pass

# Run the tests using pytest
if __name__ == '__main__':
    pytest.main()