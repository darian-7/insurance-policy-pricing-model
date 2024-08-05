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

    # One-Hot Encoding for binary categorical features
    data = pd.get_dummies(data, columns=['sex', 'smoker'], drop_first=True)

    # Convert bool to int64
    data['sex_male'] = data['sex_male'].astype('int64')
    data['smoker_yes'] = data['smoker_yes'].astype('int64')

    # Label Encoding for ordinal relationship
    label_encoder = LabelEncoder()
    data['region'] = label_encoder.fit_transform(data['region'])

    # Save encoded data to CSV
    encoded_data_path = os.path.join(output_dir, 'encoded-data.csv')
    data.to_csv(encoded_data_path, index=False)

    # Split the data into features and target
    X = data.drop(columns=['expenses'])
    y = data['expenses']

    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Concatenate the features and target variable for train and validation sets
    train = pd.concat([X_train, y_train], axis=1)
    validation = pd.concat([X_test, y_test], axis=1)

    # Ensure the data directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the train and validation data to CSV files
    train_data_path = os.path.join(output_dir, 'train.csv')
    validation_data_path = os.path.join(output_dir, 'validation.csv')
    train.to_csv(train_data_path, index=False)
    validation.to_csv(validation_data_path, index=False)

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
        s3.upload_fileobj(data_file, bucket_name, 'encoded_data/encoded-data.csv')

    print("Preprocessed dataset uploaded successfully to S3")

# Example usage
preprocess_data(bucket_name=bucket_name, file_key=file_key, output_dir='data')