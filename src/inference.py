# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import boto3
import pandas as pd
import io
import os
import joblib
import logging
import warnings
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import sagemaker

from sklearn.preprocessing import LabelEncoder
from sagemaker.model import Model
from sagemaker.serializers import CSVSerializer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_percentage_error
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# AWS code -----------------------------------------------------------------------------------------------

# Load the .env file
# load_dotenv()

# app = Flask(__name__)

# Fetch AWS credentials from environment variables
aws_access_key_id = 'AKIAQGYBPPXWQZ3AIWYS'
aws_secret_access_key = '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'
aws_region = 'eu-north-1'
role = 'arn:aws:iam::014498627053:role/sagemaker-eks-deployment'


# Initialize a session using Amazon S3
s3 = boto3.client('s3', region_name=aws_region, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

bucket_name = 'health-ins-bucket'
file_key = 'data/inference-data.csv'

obj = s3.get_object(Bucket=bucket_name, Key=file_key)
inference_data = obj['Body'].read().decode('utf-8')

sagemaker_session = sagemaker.Session(boto3.Session(region_name="eu-north-1"))

# # Load the model for Flask health check
# model_path = '/Users/darian/Desktop/C5i\ docs/C5i\ Code/insurance-policy-pricing-model/src/optimal-model-rfr.pkl'

# if model_path:
#     model = joblib.load(model_path)
# else:
#     raise ValueError("MODEL_PATH environment variable not set")

# @app.route('/ping', methods=['GET'])
# def ping():
#     return jsonify(status='ok')

# @app.route('/invocations', methods=['POST'])
# def invoke():
#     data = request.json['instances']
#     predictions = model.predict(data)
#     return jsonify(predictions=predictions.tolist())

model = Model(
    image_uri='014498627053.dkr.ecr.eu-north-1.amazonaws.com/insurance-policy-pricing-model:latest',
    role=role,
    model_data='s3://health-ins-bucket/models/optimal-model-rfr.tar.gz',
    sagemaker_session=sagemaker.Session()
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    serializer=CSVSerializer()
)

# EDA code -----------------------------------------------------------------------------------------------

# Load the data into a DataFrame
data = pd.read_csv(io.StringIO(inference_data))

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

encoded_inference_data = pd.DataFrame(data)
print(encoded_inference_data.head())

# Check the datatypes to ensure proper encoding
print(encoded_inference_data.dtypes)

# Convert the DataFrame to csv format in memory
csv_buffer = io.StringIO()
encoded_inference_data.to_csv(csv_buffer, index=False)

# Upload the csv to S3
s3.put_object(Bucket=bucket_name, Key='encoded_data/encoded-inference-data.csv', Body=csv_buffer.getvalue())

# Download the trained model from S3
model_key = 'models/optimal-model-rfr.pkl'
model_obj = s3.get_object(Bucket=bucket_name, Key=model_key)
model_data = model_obj['Body'].read()

# Load the model
model = joblib.load(io.BytesIO(model_data))

# Apply the model to the preprocessed inference data
predictions = model.predict(encoded_inference_data)

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions, columns=['predicted_expenses'])

print(predictions_df)


# MLflow code -----------------------------------------------------------------------------------------------

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = mean_absolute_percentage_error(actual, pred) * 100
    evs = explained_variance_score(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mape, r2, evs

if __name__ == "__main__":

    # Flask health check
    # app.run(host='0.0.0.0', port=8080) 

    # warnings.filterwarnings("ignore")
    # np.random.seed(40)
    
    # Split the data into training and test sets. (0.75, 0.25) split.
    X = encoded_inference_data.drop(columns=['expenses'])
    y = encoded_inference_data['expenses']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle command line arguments or set default values
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else None
    min_samples_leaf = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    min_samples_split = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    # Set the tracking URI for MLflow
    mlflow.set_tracking_uri("file:///tmp/mlruns")

    with mlflow.start_run():
        pipeline = Pipeline(steps=[
            ('regressor', RandomForestRegressor(
                max_depth=max_depth, 
                min_samples_leaf=min_samples_leaf, 
                min_samples_split=min_samples_split, 
                n_estimators=n_estimators))
        ])

        pipeline.fit(X_train, y_train)

        predicted_qualities = pipeline.predict(X_test)

        (rmse, mape, r2, evs) = eval_metrics(y_test, predicted_qualities)

        print("RFR model (n_estimators=%d, max_depth=%s, min_samples_leaf=%d, min_samples_split=%d):" % (n_estimators, max_depth, min_samples_leaf, min_samples_split))
        print("  RMSE: %s" % rmse)
        print("  MAPE: %s" % mape)
        print("  R2: %s" % r2)
        print("  Explained Variance Score: %s" % evs)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("evs", evs)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            mlflow.sklearn.log_model(pipeline, "model", registered_model_name="RFR-health-insurance-inference-data")
        else:
            mlflow.sklearn.log_model(pipeline, "model")