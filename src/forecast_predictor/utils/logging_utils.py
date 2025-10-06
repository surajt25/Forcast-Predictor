import logging
from pathlib import Path

LOG_FORMAT = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'

_initialized = False


def get_logger(name: str = 'forecast_predictor', level: int = logging.INFO) -> logging.Logger:
    global _initialized
    if not _initialized:
        logging.basicConfig(level=level, format=LOG_FORMAT)
        _initialized = True
    return logging.getLogger(name)
