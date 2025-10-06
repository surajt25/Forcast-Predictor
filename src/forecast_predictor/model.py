from dataclasses import dataclass
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
from pathlib import Path
from .config import MODEL_DIR, model_config
from .utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class ModelArtifacts:
    model_path: Path
    metrics: dict


def train_model(df: pd.DataFrame) -> ModelArtifacts:
    logger.info('Training model')
    target = model_config.target
    work = df.copy()
    # Encode any remaining object columns via category codes (simple, deterministic per run order)
    obj_cols = [c for c in work.columns if work[c].dtype == 'O' and c != target]
    for c in obj_cols:
        work[c] = work[c].astype('category').cat.codes
    # Drop datetime columns (except could extract timestamp if desired)
    datetime_cols = [c for c in work.columns if pd.api.types.is_datetime64_any_dtype(work[c])]
    feature_cols: List[str] = [
        c for c in work.columns
        if c != target
        and c not in datetime_cols
    ]
    logger.info(f"Using {len(feature_cols)} feature columns after encoding: {feature_cols[:15]}{'...' if len(feature_cols)>15 else ''}")
    X = work[feature_cols]
    y = work[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=model_config.test_size, random_state=model_config.random_state
    )

    model = RandomForestRegressor(n_estimators=200, random_state=model_config.random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    metrics = {
        'MAE': float(mean_absolute_error(y_test, preds)),
        'RMSE': rmse,
        'R2': float(r2_score(y_test, preds))
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / 'random_forest.joblib'
    joblib.dump(model, model_path)
    # Persist feature columns for inference reproducibility
    feature_meta_path = MODEL_DIR / 'random_forest.features.json'
    with open(feature_meta_path, 'w', encoding='utf-8') as f:
        json.dump({'feature_names': feature_cols}, f, indent=2)
    logger.info(f'Persisted feature metadata to {feature_meta_path}')
    logger.info(f'Model saved to {model_path} | Metrics: {metrics}')
    return ModelArtifacts(model_path=model_path, metrics=metrics)
