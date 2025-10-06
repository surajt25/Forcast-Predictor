from pathlib import Path
import pandas as pd
from typing import List
from .config import RAW_DATA_PATH
from .utils.logging_utils import get_logger

logger = get_logger(__name__)

FALLBACK_ENCODINGS: List[str] = [
    'utf-8',        # standard
    'utf-8-sig',    # handles BOM
    'cp1252',       # common Windows western
    'latin1',       # very permissive
]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV with robust encoding fallback.

    Tries a sequence of encodings until one succeeds. Logs the encoding used.
    Raises the last exception if all attempts fail.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please add 'Sample - Superstore.csv' to data/raw/"
        )
    logger.info(f"Loading raw data from {path}")
    if path.suffix.lower() != '.csv':
        raise ValueError('Unsupported file format')

    last_err: Exception | None = None
    for enc in FALLBACK_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info(f"Loaded file successfully using encoding='{enc}' | Shape={df.shape}")
            return df
        except UnicodeDecodeError as e:
            logger.warning(f"Failed decoding with encoding='{enc}': {e}")
            last_err = e
        except Exception as e:  # other read_csv errors
            logger.warning(f"General error with encoding='{enc}': {e}")
            last_err = e

    # If we reach here, all encodings failed
    raise RuntimeError(
        f"Unable to decode file {path} with tried encodings {FALLBACK_ENCODINGS}. Last error: {last_err}"
    )
