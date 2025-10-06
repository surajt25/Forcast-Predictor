import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from .config import FIGURES_DIR, model_config
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


def basic_profile(df: pd.DataFrame) -> dict:
    logger.info('Generating basic profile statistics')
    desc = df.describe(include='all').T
    missing = df.isna().sum()
    profile = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'describe': desc,
        'missing': missing[missing > 0].to_dict()
    }
    return profile


def save_fig(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{name}.png"
    fig.savefig(out, bbox_inches='tight')
    logger.info(f'Saved figure to {out}')


def eda_plots(df: pd.DataFrame):
    logger.info('Creating EDA plots')
    # Distribution of target
    if model_config.target in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df[model_config.target], kde=True, ax=ax)
        ax.set_title(f'Distribution of {model_config.target}')
        save_fig(fig, 'target_distribution')
        plt.close(fig)

    # Sales over time if date column present
    if model_config.date_col in df.columns:
        temp = df.copy()
        temp[model_config.date_col] = pd.to_datetime(temp[model_config.date_col], errors='coerce')
        ts = temp.dropna(subset=[model_config.date_col]).set_index(model_config.date_col).resample('W')[model_config.target].sum()
        fig, ax = plt.subplots(figsize=(10,4))
        ts.plot(ax=ax)
        ax.set_title('Weekly Sales Over Time')
        save_fig(fig, 'weekly_sales')
        plt.close(fig)

    # Correlation heatmap numeric features
    num_df = df.select_dtypes(include='number')
    if num_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(num_df.corr(), cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap')
        save_fig(fig, 'correlation_heatmap')
        plt.close(fig)
