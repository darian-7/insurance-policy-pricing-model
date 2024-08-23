# Import libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import boto3
import sagemaker
import tarfile
import joblib
import sklearn
import mlflow.pyfunc

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn import linear_model, tree
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import uniform
from sagemaker.serializers import CSVSerializer
from sagemaker.model import Model
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn
from time import gmtime, strftime
from scipy.stats import ks_2samp
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset

data = pd.read_csv('../data/encoded-data.csv')

# Split the data into features and target
X = data.drop(columns=['expenses'])
y = data['expenses']

# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Specify the model name and version or stage
model_name = "RFR-health-insurance"
model_stage_or_version = "Production"  # Can also use version number, e.g., '1'

# Load the model from the registry
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage_or_version}")

model.fit(X_train, y_train)

# Predictions and evaluation
pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
evs = explained_variance_score(y_test, pred)
print(f'RMSE: {rmse}')
print(f'R^2: {r2}')
print(f'Explained Variance Score: {evs}')

# Cross-validation performance metrics
cv_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
cv_evs = cross_val_score(model, X_train, y_train, cv=5, scoring='explained_variance')

print('Cross-Validation Performance metrics')
print(f'CV RMSE: {cv_rmse.mean()}')
print(f'CV R^2: {cv_r2.mean()}')
print(f'CV Explained Variance Score: {cv_evs.mean()}')

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

print("Model successfully trained")