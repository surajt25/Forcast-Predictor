from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_PATH = DATA_DIR / 'raw' / 'Sample - Superstore.csv'
PROCESSED_DATA_PATH = DATA_DIR / 'processed' / 'superstore_processed.parquet'
MODEL_DIR = BASE_DIR / 'models'
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'

@dataclass(frozen=True)
class ModelConfig:
    target: str = 'Sales'
    date_col: str = 'Order Date'
    id_col: str = 'Order ID'
    test_size: float = 0.2
    random_state: int = 42

model_config = ModelConfig()
