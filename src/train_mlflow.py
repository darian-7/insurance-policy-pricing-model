import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

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
    min_samples_leaf = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    min_samples_split = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    # Set the tracking URI for MLflow
    mlflow.set_tracking_uri("file:///tmp/mlruns")

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

        print("RFR model (n_estimators=%d, min_samples_leaf=%d, min_samples_split=%d):" % (n_estimators, min_samples_leaf, min_samples_split))
        print("  RMSE: %s" % rmse)
        print("  R2: %s" % r2)
        print("  Explained Variance Score: %s" % evs)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("evs", evs)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            mlflow.sklearn.log_model(pipeline, "model", registered_model_name="RFR-health-insurance")
        else:
            mlflow.sklearn.log_model(pipeline, "model")
