from pathlib import Path
import pandas as pd
from .data_loader import load_raw_data
from .eda import basic_profile, eda_plots
from .features import engineer_features
from .model import train_model
from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_pipeline():
    logger.info('Starting pipeline')
    df = load_raw_data(RAW_DATA_PATH)
    profile = basic_profile(df)
    eda_plots(df)
    feat_df = engineer_features(df)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(PROCESSED_DATA_PATH, index=False)
    artifacts = train_model(feat_df)
    return {
        'profile_columns': profile['columns'],
        'metrics': artifacts.metrics,
        'model_path': str(artifacts.model_path),
    }

if __name__ == '__main__':
    results = run_pipeline()
    print(results)
