import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys
import plotly.express as px
import numpy as np

# Ensure src directory on path when running via `python app/streamlit_app.py` or Streamlit CLI
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecast_predictor.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, MODEL_DIR, model_config
from forecast_predictor.data_loader import load_raw_data
from forecast_predictor.features import engineer_features, make_inference_frame
from forecast_predictor.model import train_model

st.set_page_config(page_title='Forecast Predictor', layout='wide', page_icon='📈')

st.markdown("""
# 📈 Forecast Predictor Dashboard
Gain insights into Superstore sales data: explore, engineer features, and generate forecasts.
""")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header('🧭 Controls')
    raw = None  # lazy load placeholder
    date_filter_enabled = st.checkbox('Enable Date Range Filter', value=False)
    segment_filter_enabled = st.checkbox('Filter by Segment', value=False)
    category_filter_enabled = st.checkbox('Filter by Category', value=False)
    sample_size = st.slider('Prediction Sample Size', 5, 100, 20, step=5)
    st.markdown('---')
    st.caption('Tip: Use the tabs to navigate between raw data, features, and model predictions.')

@st.cache_data
def get_raw():
    return load_raw_data(RAW_DATA_PATH)

@st.cache_data
def get_processed():
    if PROCESSED_DATA_PATH.exists():
        return pd.read_parquet(PROCESSED_DATA_PATH)
    raw = get_raw()
    return engineer_features(raw)

@st.cache_resource
def get_model_and_features():
    model_path = MODEL_DIR / 'random_forest.joblib'
    feature_meta = MODEL_DIR / 'random_forest.features.json'
    if model_path.exists() and feature_meta.exists():
        model = joblib.load(model_path)
        import json
        meta = json.loads(feature_meta.read_text())
        feature_names = meta.get('feature_names', [])
        return model, feature_names
    # Train if not present
    df = get_processed()
    artifacts = train_model(df)
    import json
    meta = json.loads((MODEL_DIR / 'random_forest.features.json').read_text())
    return joblib.load(artifacts.model_path), meta.get('feature_names', [])

raw_tab, features_tab, eda_tab, model_tab = st.tabs(["📄 Raw Data", "🛠 Features", "🧐 EDA", "🤖 Model & Forecast"])

with raw_tab:
    st.subheader('Raw Data Preview')
    raw = get_raw()
    df_display = raw.copy()
    # Filters
    if date_filter_enabled and model_config.date_col in df_display.columns:
        df_display[model_config.date_col] = pd.to_datetime(df_display[model_config.date_col], errors='coerce')
        min_d, max_d = df_display[model_config.date_col].min(), df_display[model_config.date_col].max()
        start, end = st.date_input('Date Range', value=(min_d.date(), max_d.date()))
        df_display = df_display[(df_display[model_config.date_col] >= pd.to_datetime(start)) & (df_display[model_config.date_col] <= pd.to_datetime(end))]
    if segment_filter_enabled and 'Segment' in df_display.columns:
        segs = st.multiselect('Segment', sorted(df_display['Segment'].unique()), default=list(df_display['Segment'].unique()))
        df_display = df_display[df_display['Segment'].isin(segs)]
    if category_filter_enabled and 'Category' in df_display.columns:
        cats = st.multiselect('Category', sorted(df_display['Category'].unique()), default=list(df_display['Category'].unique()))
        df_display = df_display[df_display['Category'].isin(cats)]

    st.dataframe(df_display.head(50))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Rows', f"{len(df_display):,}")
    with col2:
        st.metric('Total Sales', f"${df_display[model_config.target].sum():,.0f}" if model_config.target in df_display else '—')
    with col3:
        if 'Profit' in df_display:
            margin = (df_display['Profit'].sum() / df_display[model_config.target].sum()) if model_config.target in df_display and df_display[model_config.target].sum() else 0
            st.metric('Profit Margin', f"{margin:.1%}")
        else:
            st.metric('Profit Margin', '—')
    with col4:
        if 'Discount' in df_display:
            st.metric('Avg Discount', f"{df_display['Discount'].mean():.2f}")
        else:
            st.metric('Avg Discount', '—')

with features_tab:
    st.subheader('Engineered Features Sample')
    feat = get_processed()
    st.dataframe(feat.head(30))
    st.caption(f"Engineered feature set shape: {feat.shape}")

with eda_tab:
    st.subheader('Exploratory Data Analysis')
    figures_dir = ROOT / 'reports' / 'figures'
    fig_files = {
        'Target Distribution': figures_dir / 'target_distribution.png',
        'Weekly Sales': figures_dir / 'weekly_sales.png',
        'Correlation Heatmap': figures_dir / 'correlation_heatmap.png'
    }
    cols = st.columns(3)
    for (title, path), col in zip(fig_files.items(), cols):
        with col:
            if path.exists():
                st.image(str(path), caption=title, width='stretch')
            else:
                st.info(f"{title} not generated yet. Run training pipeline.")

    raw_full = get_raw()

    # Region Sales Bar Chart
    bar_fig = None
    if 'Region' in raw_full.columns and model_config.target in raw_full.columns:
        region_sales = (raw_full.groupby('Region')[model_config.target]
                                 .sum()
                                 .sort_values(ascending=False)
                                 .reset_index())
        bar_fig = px.bar(region_sales, x='Region', y=model_config.target,
                          title='Sales by Region', text_auto='.2s', color='Region')
        bar_fig.update_layout(showlegend=False, xaxis_title='', yaxis_title='Total Sales')

    # Weekly Sales Line Chart
    weekly_fig = None
    if model_config.date_col in raw_full.columns:
        temp = raw_full.copy()
        temp[model_config.date_col] = pd.to_datetime(temp[model_config.date_col], errors='coerce')
        weekly = (temp.dropna(subset=[model_config.date_col])
                    .set_index(model_config.date_col)
                    .resample('W')[model_config.target]
                    .sum()
                    .reset_index())
        weekly_fig = px.line(weekly, x=model_config.date_col, y=model_config.target, title='Weekly Sales (Interactive)')

    # Pie Chart Selector
    pie_dims = [d for d in ['Region', 'Category', 'Segment', 'Sub-Category'] if d in raw_full.columns]

    st.markdown('### Comparative Visuals')
    c1, c2 = st.columns(2)
    with c1:
        if bar_fig is not None:
            st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.info('Region data not available.')
        if weekly_fig is not None:
            st.plotly_chart(weekly_fig, use_container_width=True)
        else:
            st.info('Date column not available for weekly aggregation.')
    with c2:
        if pie_dims:
            pie_dim = st.selectbox('Pie Dimension', pie_dims, index=0)
            pie_df = raw_full.groupby(pie_dim)[model_config.target].sum().reset_index()
            pie_fig = px.pie(pie_df, names=pie_dim, values=model_config.target, title=f'Sales Distribution by {pie_dim}')
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info('No categorical dimensions available for pie chart.')

with model_tab:
    st.subheader('Model & Predictions')
    model, trained_features = get_model_and_features()
    raw_full = get_raw()
    colA, colB = st.columns([2,1])
    with colA:
        if st.button('Generate Prediction Sample'):
            aligned = make_inference_frame(raw_full.sample(min(len(raw_full), sample_size*3), random_state=42), trained_features)
            sample = aligned.sample(sample_size, random_state=1)
            preds = model.predict(sample)
            sample = sample.assign(PredictedSales=preds)
            st.dataframe(sample.head(sample_size))
    with colB:
        # Feature importances
        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(model.feature_importances_, index=trained_features).sort_values(ascending=False)[:15]
            imp_fig = px.bar(importances[::-1], orientation='h', title='Top Feature Importances')
            st.plotly_chart(imp_fig, use_container_width=True)
        else:
            st.info('Feature importances unavailable for this model.')

st.markdown('---')
st.caption('Forecast Predictor © 2025 · Built with Streamlit, scikit-learn, and Plotly')
