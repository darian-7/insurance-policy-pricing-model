import os
import warnings
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import logging
import mlflow.pyfunc
import boto3
import joblib


from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from urllib.parse import urlparse

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    evs = explained_variance_score(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, r2, evs

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the csv file
    data = pd.read_csv("../data/encoded-data.csv")
    
    # Split the data into training and test sets. (0.75, 0.25) split.
    X = data.drop(columns=['expenses'])
    y = data['expenses']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle command line arguments or set default values
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    min_samples_leaf = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    min_samples_split = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    # Set the tracking URI for MLflow to your server
    mlflow.set_tracking_uri("http://localhost:5002")

    # Check if experiment exists; if not, create it
    experiment_name = "Health_Insurance_RFR_Model_28"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    else:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        pipeline = Pipeline(steps=[
            ('regressor', RandomForestRegressor(
                min_samples_leaf=min_samples_leaf, 
                min_samples_split=min_samples_split, 
                n_estimators=n_estimators))
        ])

        pipeline.fit(X_train, y_train)

        predicted_qualities = pipeline.predict(X_test)

        (rmse, r2, evs) = eval_metrics(y_test, predicted_qualities)

        print(f"  RFR model (n_estimators={n_estimators}, min_samples_leaf={min_samples_leaf}, min_samples_split={min_samples_split}):")
        print(f"  RMSE: {rmse}")
        print(f"  R2: {r2}")
        print(f"  Explained Variance Score: {evs}")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("evs", evs)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            mlflow.sklearn.log_model(pipeline, "model", registered_model_name="RFR-health-ins")
        else:
            mlflow.sklearn.log_model(pipeline, "model")

    model_name = "RFR-health-ins"
    model_stage_or_version = "7"  # Or use 'Production' or a specific version, e.g., '7'

    # Load the model from the registry
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage_or_version}")

    # Define the local path to save the model
    local_model_path = f"/models/{model_name}.pkl"

    # Save the model locally using joblib
    joblib.dump(model, local_model_path)

    # Define your S3 bucket name and the key where the model will be stored
    s3_bucket_name = "health-ins-bucket"
    s3_model_key = f"models/{model_name}/{model_stage_or_version}/{model_name}.pkl"

    # Upload the model to S3 using boto3
    s3_client = boto3.client('s3')
    s3_client.upload_file(local_model_path, s3_bucket_name, s3_model_key)

    print(f"Model saved to S3 at s3://{s3_bucket_name}/{s3_model_key}")