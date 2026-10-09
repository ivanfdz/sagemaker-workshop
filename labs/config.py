"""Configuración común de los labs. Rellena los <...> antes de empezar."""
import time

import boto3

REGION = "eu-west-1"
ROLE_ARN = "arn:aws:iam::<account-id>:role/<SageMakerExecutionRole>"  # rol que asume SageMaker
BUCKET = "<bucket-ml>"                                                 # misma región que REGION
BASE = "ml/kwh-prediction"
DATASET = f"{BASE}/datasets/v=2026-10-08"                              # versión inmutable del dataset
TS = time.strftime("%Y%m%d-%H%M%S")                                    # sufijo único por ejecución

# True: genera sesiones de carga sintéticas (no necesitas tabla gold).
# False: lee de Athena con la query del Lab 1.
USE_SYNTHETIC = True
ATHENA_DATABASE = "gold"

sm = boto3.client("sagemaker", region_name=REGION)
smr = boto3.client("sagemaker-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
aas = boto3.client("application-autoscaling", region_name=REGION)
