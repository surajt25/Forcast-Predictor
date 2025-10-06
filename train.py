import sys
from pathlib import Path

# Ensure src directory is on path when running as script
ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecast_predictor.pipeline import run_pipeline

if __name__ == '__main__':
    out = run_pipeline()
    print(out)
