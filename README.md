# Forecast Predictor

An end-to-end data science project for sales forecasting using the Sample Superstore dataset.

## Project Structure
```
├── data
│   ├── raw/                # Place original dataset here (Sample - Superstore.csv)
│   └── processed/          # Engineered / cleaned data artifacts
├── models/                 # Saved trained models
├── notebooks/              # Jupyter notebooks for experimentation
├── reports/
│   └── figures/            # Generated EDA figures
├── src/
│   └── forecast_predictor/
│       ├── config.py
│       ├── data_loader.py
│       ├── eda.py
│       ├── features.py
│       ├── model.py
│       ├── pipeline.py
│       └── utils/
│           └── logging_utils.py
├── app/
│   └── streamlit_app.py    # Streamlit dashboard
├── train.py                # Runs pipeline
├── requirements.txt
└── README.md
```

## Getting Started
1. Create & activate a virtual environment (optional but recommended):
```
python -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Add dataset: Place `Sample - Superstore.csv` into `data/raw/`.
4. Run training pipeline:
```
python -m train
```
5. Launch Streamlit app:
```
streamlit run app/streamlit_app.py
```

## Pipeline Steps
1. Load raw data
2. Generate basic profiling & EDA plots
3. Feature engineering (date parts, lags, categorical encoding)
4. Train RandomForest model
5. Save model & metrics

## Outputs
- Engineered dataset: `data/processed/superstore_processed.parquet`
- Model: `models/random_forest.joblib`
- Figures: `reports/figures/*.png`

## Customization
Adjust target, date column, and parameters in `src/forecast_predictor/config.py`.

## Next Ideas
- Add time series specific models (Prophet, XGBoost)
- Hyperparameter tuning
- Model registry and versioning
- CI pipeline for tests & linting

Enjoy forecasting! 🚀
