import pandas as pd
from typing import List
from .config import model_config
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Engineering features')
    out = df.copy()
    # Parse dates
    if model_config.date_col in out.columns:
        out[model_config.date_col] = pd.to_datetime(out[model_config.date_col], errors='coerce')
        out['Year'] = out[model_config.date_col].dt.year
        out['Month'] = out[model_config.date_col].dt.month
        out['WeekOfYear'] = out[model_config.date_col].dt.isocalendar().week.astype(int)
    # Simple lag feature (aggregate by week)
    if model_config.date_col in out.columns and model_config.target in out.columns:
        weekly = (out.dropna(subset=[model_config.date_col])
                    .set_index(model_config.date_col)
                    .resample('W')[model_config.target]
                    .sum()
                    .to_frame('WeeklySales'))
        weekly['WeeklySalesLag1'] = weekly['WeeklySales'].shift(1)
        weekly['WeeklySalesLag2'] = weekly['WeeklySales'].shift(2)
        weekly = weekly.reset_index().rename(columns={model_config.date_col: 'WeekDate'})
        out = out.merge(weekly, left_on=out[model_config.date_col].dt.to_period('W').astype(str),
                        right_on=weekly['WeekDate'].dt.to_period('W').astype(str), how='left')
        out.drop(columns=['key_0'], errors='ignore', inplace=True)
    # Encode categoricals (simple)
    for col in out.select_dtypes(include='object').columns:
        if col != model_config.target:
            out[col] = out[col].astype('category').cat.codes
    out = out.dropna(subset=[model_config.target])
    return out


def make_inference_frame(raw_df: pd.DataFrame, trained_feature_names: List[str]) -> pd.DataFrame:
    """Create a feature dataframe aligned to training feature columns.

    This re-applies feature engineering and then:
      * ensures categorical encoding consistency (using category codes again)
      * drops datetime columns
      * reindexes to the feature list used at training, adding any missing cols (filled with 0)
    """
    feat = engineer_features(raw_df.copy())
    # Remove target if still present among features
    if model_config.target in feat.columns:
        pass  # kept for potential evaluation; not removed yet
    # Encode any object columns
    for c in feat.select_dtypes(include='object').columns:
        if c != model_config.target:
            feat[c] = feat[c].astype('category').cat.codes
    # Drop datetime columns
    for c in list(feat.columns):
        if pd.api.types.is_datetime64_any_dtype(feat[c]):
            feat = feat.drop(columns=[c])
    # Keep only trained feature names intersection, then add missing
    present = [c for c in trained_feature_names if c in feat.columns]
    missing = [c for c in trained_feature_names if c not in feat.columns]
    for m in missing:
        feat[m] = 0
    aligned = feat[trained_feature_names]
    return aligned
