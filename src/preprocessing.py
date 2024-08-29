import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import boto3
import io
import os

# Fetch AWS credentials from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Initialize a session using Amazon S3
s3 = boto3.client('s3', region_name='eu-north-1', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

bucket_name = 'health-ins-bucket'
file_key = 'data/health-insurance.csv'

def preprocess_data(bucket_name, file_key, output_dir):
    # Download raw data from S3
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    raw_data = obj['Body'].read().decode('utf-8')
    
    # Load the data into a DataFrame
    data = pd.read_csv(io.StringIO(raw_data))

    # Print the first few rows of the data to verify it was loaded correctly
    print("Initial data:")
    print(data.head())

    # Label Encoding for binary categorical features
    label_encoder = LabelEncoder()
    data['sex'] = label_encoder.fit_transform(data['sex'])
    data['smoker'] = label_encoder.fit_transform(data['smoker'])

    # Print the columns to check if encoding was successful
    print("Data columns after label encoding:")
    print(data.columns)

    # Convert the encoded columns to int64
    data['sex'] = data['sex'].astype('int64')
    data['smoker'] = data['smoker'].astype('int64')

    # Label Encoding for ordinal relationship in 'region'
    data['region'] = label_encoder.fit_transform(data['region'])

    # Ensure the data directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the encoded data to CSV
    encoded_data_path = os.path.join(output_dir, 'encoded-data.csv')
    data.to_csv(encoded_data_path, index=False)

    # Plot histograms
    distribution_plots = ['bmi', 'expenses']
    for feature in distribution_plots:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[feature], kde=True, bins=30)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()

    # Visualize correlation matrix
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

    print(data.describe())
    print(data.head())
    print(data.dtypes)

    # Upload encoded data to S3
    with open(encoded_data_path, 'rb') as data_file:
        s3.upload_fileobj(data_file, bucket_name, 'data/encoded-data.csv')

    print(f"Preprocessed dataset uploaded successfully to S3 at s3://{bucket_name}/data/encoded-data.csv")

# Example usage
preprocess_data(bucket_name=bucket_name, file_key=file_key, output_dir='data')