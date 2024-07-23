# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import boto3
import io
import os
import mlflow
import mlflow.sklearn
import logging
import warnings
import sys
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from urllib.parse import urlparse


# Fetch AWS credentials from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Initialize a session using Amazon S3
s3 = boto3.client('s3', region_name='eu-north-1', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

bucket_name = 'health-insurance-bucket'
file_key = 'encoded_data/encoded-data.csv'

obj = s3.get_object(Bucket=bucket_name, Key=file_key)
raw_data = obj['Body'].read().decode('utf-8')

# Load the data into a DataFrame
encoded_data = pd.read_csv(io.StringIO(raw_data))

# Define features and target variable
X = encoded_data.drop(columns=['expenses'])
y = encoded_data['expenses']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection after comparing common regression models, params selected after GridSearchCV
model = RandomForestRegressor(max_depth = None, min_samples_leaf = 4, min_samples_split = 10, n_estimators = 600)

model.fit(X_train, y_train)

# Save the trained model to a local file
model_filename = 'optimal-model-rfr.pkl'
joblib.dump(model, model_filename)

# Upload the model file to S3
s3.upload_file(model_filename, bucket_name, f'models/{model_filename}')

pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
evs = explained_variance_score(y_test, pred)
mape = mean_absolute_percentage_error(y_test, pred) * 100

# Evaluate model using cross-validation
cv_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
cv_evs = cross_val_score(model, X_train, y_train, cv=5, scoring='explained_variance')
cv_mape = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_percentage_error')

print('Cross-Validation Performance metrics')
print(f'CV RMSE: {cv_rmse.mean()}')
print(f'CV R^2: {cv_r2.mean()}')
print(f'CV Explained Variance Score: {cv_evs.mean()}')
print(f'CV MAPE: {-cv_mape.mean() * 100}%')

print('Performance metrics')
print(f'RMSE: {rmse}')
print(f'R^2: {r2}')
print(f'Explained Variance Score: {evs}')
print(f'MAPE: {mape}%')

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


# MLflow code ------------------------------------------------------------------------

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = mean_absolute_percentage_error(actual, pred) * 100
    evs = explained_variance_score(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mape, r2, evs

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # Split the data into training and test sets. (0.75, 0.25) split.
    X = encoded_data.drop(columns=['expenses'])
    y = encoded_data['expenses']
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
            mlflow.sklearn.log_model(pipeline, "model", registered_model_name="RFR-health-insurance")
        else:
            mlflow.sklearn.log_model(pipeline, "model")