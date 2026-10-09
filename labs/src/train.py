"""Script de training para el Lab 3 (script mode, contenedor sklearn de SageMaker).

Lee Parquet de los canales train/validation, entrena un regresor y guarda
el modelo en SM_MODEL_DIR, que SageMaker empaqueta como model.tar.gz.
"""
import argparse
import glob
import os

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

TARGET = "energy_kwh"


def read_channel(name: str) -> pd.DataFrame:
    """Lee todos los .parquet de un canal (/opt/ml/input/data/<name>)."""
    path = os.environ.get(f"SM_CHANNEL_{name.upper()}", f"/opt/ml/input/data/{name}")
    files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No hay ficheros parquet en {path}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Los hiperparámetros del ModelTrainer llegan como --nombre valor
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args, _ = parser.parse_known_args()

    train, val = read_channel("train"), read_channel("validation")
    y_train, X_train = train.pop(TARGET), train
    y_val, X_val = val.pop(TARGET), val
    print(f"train={len(X_train)} filas, validation={len(X_val)} filas, features={list(X_train.columns)}")

    model = HistGradientBoostingRegressor(max_iter=args.max_iter, learning_rate=args.learning_rate)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_val, model.predict(X_val))
    # Formato fijo para poder capturarlo con regex (métricas / HPO): val_mae=([0-9\.]+)
    print(f"val_mae={mae:.4f}")

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
