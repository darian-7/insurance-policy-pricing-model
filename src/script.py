
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split, cross_val_score
from joblib import dump, load
import io

os.chdir("/Users/darian/Desktop/C5i docs/C5i Code/insurance-policy-pricing-model/src")

# # Functions required for SageMaker
# def model_fn(model_dir):
#     """Load model from the model_dir."""
#     model = load(os.path.join(model_dir, "model.joblib"))
#     return model

# def input_fn(input_data, content_type):
#     """Deserialize the input data."""
#     if content_type == "text/csv":
#         return pd.read_csv(io.StringIO(input_data), header=None)
#     else:
#         raise ValueError("Unsupported content type: {}".format(content_type))

# def predict_fn(input_data, model):
#     """Make predictions using the model."""
#     return model.predict(input_data)

# def output_fn(prediction, content_type):
#     """Serialize the output data."""
#     if content_type == "text/csv":
#         return ",".join(map(str, prediction))
#     else:
#         raise ValueError("Unsupported content type: {}".format(content_type))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--n_estimators", type=int, default=600)
#     parser.add_argument("--min_samples_split", type=int, default=10)
#     parser.add_argument("--min_samples_leaf", type=int, default=4)
#     parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
#     parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
#     args = parser.parse_args()
    
#     # Load the datasets
#     train_df = pd.read_csv(os.path.join(args.train, "train.csv"))
#     validation_df = pd.read_csv(os.path.join(args.validation, "validation.csv"))
    
#     # Split features and target
#     X_train = train_df.drop(columns=['expenses'])
#     y_train = train_df['expenses']
#     X_validation = validation_df.drop(columns=['expenses'])
#     y_validation = validation_df['expenses']
    
#     # Initialize and train the model
#     model = RandomForestRegressor(
#         n_estimators=args.n_estimators,
#         min_samples_split=args.min_samples_split,
#         min_samples_leaf=args.min_samples_leaf
#     )
#     model.fit(X_train, y_train)
    
#     # Save the model
#     model_dir = os.environ.get("SM_MODEL_DIR")
#     dump(model, os.path.join(model_dir, "model.joblib"))
    
#     # Predictions and evaluation
#     pred = model.predict(X_validation)
#     rmse = np.sqrt(mean_squared_error(y_validation, pred))
#     r2 = r2_score(y_validation, pred)
#     evs = explained_variance_score(y_validation, pred)
#     print(f'RMSE: {rmse}')
#     print(f'R^2: {r2}')
#     print(f'Explained Variance Score: {evs}')
    
#     # Cross-validation performance metrics
#     cv_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
#     cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
#     cv_evs = cross_val_score(model, X_train, y_train, cv=5, scoring='explained_variance')
    
#     print('Cross-Validation Performance metrics')
#     print(f'CV RMSE: {cv_rmse.mean()}')
#     print(f'CV R^2: {cv_r2.mean()}')
#     print(f'CV Explained Variance Score: {cv_evs.mean()}')
