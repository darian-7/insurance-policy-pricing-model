import boto3
import sagemaker
import os

from sagemaker.sklearn.estimator import SKLearn
from time import gmtime, strftime

os.chdir("/Users/darian/Desktop/C5i docs/C5i Code/insurance-policy-pricing-model/src")

# Fetch AWS credentials
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = 'eu-north-1'

# Initialize a session using Amazon S3
s3 = boto3.client('s3', region_name=aws_region, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
bucket_name = 'health-ins-bucket'

FRAMEWORK_VERSION = "0.23-1"

sagemaker_session = sagemaker.Session()

sklearn_estimator = SKLearn(
    entry_point="script.py",
    role="arn:aws:iam::014498627053:role/sagemaker-eks-deployment",
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version=FRAMEWORK_VERSION,
    base_job_name="rfr-sklearn",
    hyperparameters={
        "n_estimators": 600,
        "min_samples_split": 10,
        "min_samples_leaf": 4
    }
)

train_path = 's3://health-ins-bucket/data/train.csv'
validation_path = 's3://health-ins-bucket/data/validation.csv'

sklearn_estimator.fit({"train": train_path, "validation": validation_path}, wait=True)

# Create a SageMaker client
sagemaker_client = boto3.client('sagemaker')

# Describe the training job to get the model artifacts
response = sagemaker_client.describe_training_job(
    TrainingJobName=sklearn_estimator.latest_training_job.name
)

artifact = response["ModelArtifacts"]["S3ModelArtifacts"]

print("Model artifact persisted at " + artifact)


# # AWS endpoint deployment
# sklearn_estimator = SKLearn.attach(training_job_name="rfr-sklearn-2024-08-03-09-42-50-995")

# endpoint_name = "custom-RFR-model-deploy-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
# print("EndpointName={}".format(endpoint_name))

# predictor = sklearn_estimator.deploy(
#     initial_instance_count=1,
#     instance_type="ml.m5.large",
#     endpoint_name=endpoint_name,
# )

# # Cleanup - to avoid charges
# sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

"""
CLEANUP ON AWS UI:
Inference -> Endpoints -> Actions -> Delete.
Inference -> Endpoint configurations -> Actions -> Delete.
Inference -> Models -> Actions -> Delete.
Notebook -> Notebook Instances -> Actions -> Stop (Wait) -> Actions -> Delete.
"""