"""Configuración común de los labs. Rellena los <...> antes de empezar."""
import os
import time
from pathlib import Path

import boto3

# Credenciales temporales: pega tal cual el bloque "export AWS_..." del portal de AWS en
# labs/.env (está en .gitignore). Si el fichero no existe, se usa la cadena de credenciales normal.
ENV_FILE = Path(__file__).with_name(".env")
if ENV_FILE.exists():
    for raw in ENV_FILE.read_text().splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:
            os.environ[key.strip()] = value

REGION = "eu-west-1"
# Rellena estos dos en labs/.env (no se sube a git) o cámbialos aquí por tus valores:
#   SAGEMAKER_ROLE_ARN=arn:aws:iam::<account-id>:role/<tu-rol>
#   SAGEMAKER_BUCKET=<tu-bucket>
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "arn:aws:iam::<account-id>:role/<SageMakerExecutionRole>")
BUCKET = os.environ.get("SAGEMAKER_BUCKET", "<bucket-ml>")  # misma región que REGION
BASE = "ml/kwh-prediction"
DATASET = f"{BASE}/datasets/v=2026-10-08"                              # versión inmutable del dataset
TS = time.strftime("%Y%m%d-%H%M%S")                                    # sufijo único por ejecución

# True: genera sesiones de carga sintéticas (no necesitas tabla gold).
# False: lee de Athena con la query del Lab 1.
USE_SYNTHETIC = True
ATHENA_DATABASE = "gold"

# Sesión por defecto nueva (la usan boto3 y awswrangler): al recargar config relee las credenciales
boto3.setup_default_session(region_name=REGION)
sm = boto3.client("sagemaker", region_name=REGION)
smr = boto3.client("sagemaker-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
aas = boto3.client("application-autoscaling", region_name=REGION)
