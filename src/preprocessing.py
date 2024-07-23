# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import boto3
import pandas as pd
import io
import os

# Fetch AWS credentials from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Initialize a session using Amazon S3
s3 = boto3.client('s3', region_name='eu-north-1', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

bucket_name = 'health-insurance-bucket'
file_key = 'raw_data/health-insurance.csv'

obj = s3.get_object(Bucket=bucket_name, Key=file_key)
raw_data = obj['Body'].read().decode('utf-8')

# Load the data into a DataFrame
data = pd.read_csv(io.StringIO(raw_data))

# Identify categorical features
categorical_features = ['sex', 'region', 'smoker']

# One-Hot Encoding for binary categorical features
data = pd.get_dummies(data, columns=['sex', 'smoker'], drop_first=True)
print(data)
# Convert bool to int64
data['sex_male'] = data['sex_male'].astype('int64')
data['smoker_yes'] = data['smoker_yes'].astype('int64')

# Label Encoding for ordinal relationship
label_encoder = LabelEncoder()
data['region'] = label_encoder.fit_transform(data['region'])

# Visualize correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

print(data.describe())

encoded_df = pd.DataFrame(data)
print(encoded_df.head())

# Check the datatypes to ensure proper encoding
print(encoded_df.dtypes)

# Convert the DataFrame to csv format in memory
csv_buffer = io.StringIO()
encoded_df.to_csv(csv_buffer, index=False)

# Upload the csv to S3
s3.put_object(Bucket=bucket_name, Key='encoded_data/encoded-data.csv', Body=csv_buffer.getvalue())
