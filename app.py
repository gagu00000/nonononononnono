# app.py - NovaMart Marketing Analytics Dashboard (Hybrid Theme: Dark app + Corporate Blue sidebar/charts)
# Place the 11 CSVs in the same folder as this file (root).
# Requirements: streamlit, pandas, numpy, plotly, scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import json, urllib.request

# ---------------------------
# THEME SETTINGS
# ---------------------------
APP_BG = "#0E1117"         # dark app background
TEXT_COLOR = "#FFFFFF"     # white text
SIDEBAR_BLUE = "#1F4E79"   # user choice C (soft corporate blue)
PRIMARY = "#0B3D91"
ACCENT = "#2B8CC4"
PALETTE = [PRIMARY, ACCENT, "#66A3D2", "#B2D4EE", "#F4B400"]

st.set_page_config(page_title="NovaMart Marketing Dashboard", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Plotly dark template with corporate blue palette
# ---------------------------
pio.templates["hybrid_blue"] = pio.templates["plotly_dark"]
pio.templates["hybrid_blue"].layout.update({
    "paper_bgcolor": APP_BG,
    "plot_bgcolor": APP_BG,
    "font": {"color": TEXT_COLOR, "family": "Arial"},
    "colorway": PALETTE,
    "legend": {"title_font": {"color": TEXT_COLOR}, "font": {"color": TEXT_COLOR}},
    "title": {"x": 0.01, "font": {"color": TEXT_COLOR}},
})
pio.templates.default = "hybrid_blue"

# ---------------------------
# CSS: Force dark app + blue sidebar; readable metrics & inputs
# ---------------------------
st.markdown(f"""
<style>
/* App background and main text */
body, .stApp, .block-container {{
  background-color: {APP_BG} !important;
  color: {TEXT_COLOR} !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
  background-color: {SIDEBAR_BLUE} !important;
  color: {TEXT_COLOR} !important;
}}

/* Metric card style */
div[data-testid="metric-container"] {{
  background: rgba(255,255,255,0.03) !important;
  padding: 10px !important;
  border-radius: 8px;
}}
div[data-testid="metric-container"] .stMetricLabel, div[data-testid="metric-container"] .stMetricValue {{
  color: {TEXT_COLOR} !important;
}}

/* Inputs readability */
.stSelectbox, .stTextInput, .stNumberInput, .stDateInput, .stMultiSelect, .stSlider {{
  color: {TEXT_COLOR} !important;
}}

/* Plotly container background (let template handle chart backgrounds) */
.stPlotlyChart > div {{
  background: transparent !important;
}}

/* Ensure headers and markdown show in white */
h1, h2, h3, h4, h5, p, span, label {{
  color: {TEXT_COLOR} !important;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# DATA LOADING (root)
# ---------------------------
@st.cache_data
def safe_read_csv(path, parse_dates=None):
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_all():
    files = {
        'campaign': "campaign_performance.csv",
        'customer': "customer_data.csv",
        'product': "product_sales.csv",
        'lead': "lead_scoring_results.csv",
        'feature_importance': "feature_importance.csv",
        'learning_curve': "learning_curve.csv",
        'geo': "geographic_data.csv",
        'attribution': "channel_attribution.csv",
        'funnel': "funnel_data.csv",
        'journey': "customer_journey.csv",
        'corr': "correlation_matrix.csv"
    }
    data = {}
    for k, fname in files.items():
        if k == 'campaign':
            data[k] = safe_read_csv(fname, parse_dates=['date'])
        else:
            data[k] = safe_read_csv(fname)

    # enrich campaign df
    if not data['campaign'].empty and 'date' in data['campaign'].columns:
        try:
            data['campaign']['date'] = pd.to_datetime(data['campaign']['date'])
            data['campaign']['year'] = data['campaign']['date'].dt.year
            data['campaign']['month'] = data['campaign']['date'].dt.strftime('%B')
            data['campaign']['quarter'] = data['campaign']['date'].dt.to_period('Q').astype(str)
        except Exception:
            pass

    # Normalize learning_curve column names if necessary
    if not data['learning_curve'].empty:
        lc = data['learning_curve']
        rename_map = {}
        if 'training_size' in lc.columns and 'train_size' not in lc.columns:
            rename_map['training_size'] = 'train_size'
        if 'validation_score' in lc.columns and 'val_score' not in lc.columns:
            rename_map['validation_score'] = 'val_score'
        if rename_map:
            lc = lc.rename(columns=rename_map)
            data['learning_curve'] = lc

    return data

data = load_all()

# ---------------------------
# Helpers
# ---------------------------
def df_or_warn(key):
    df = data.get(key)
    if df is None or df.empty:
        st.warning(f"Dataset `{key}` missing or empty. Upload `{key}.csv` in the repo root to enable related charts.")
        return pd.DataFrame()
    return df.copy()

def money(x):
    try:
        return f"₹{x:,.0f}"
    except Exception:
        return x

# ---------------------------
# VISUALIZATION FUNCTIONS
# ---------------------------

# Executive KPIs
def kpi_overview():
    st.header("Executive Overview")
    df = df_or_warn('campaign')
    cust = df_or_warn('customer')
    c1, c2, c3, c4 = st.columns(4)
    if df.empty:
        c1.metric("Total Revenue", "N/A")
        c2.metric("Total Conversions", "N/A")
        c3.metric("Total Spend", "N/A")
        c4.metric("ROAS", "N/A")
    else:
        total_rev = df['revenue'].sum() if 'revenue' in df.columns else 0
        total_conv = df['conversions'].sum() if 'conversions' in df.columns else 0
        total_spend = df['spend'].sum() if 'spend' in df.columns else 0
        roas = total_rev / total_spend if total_spend else np.nan
        c1.metric("Total Revenue", money(total_rev))
        c2.metric("Total Conversions", f"{int(total_conv):,}")
        c3.metric("Total Spend", money(total_spend))
        c4.metric("ROAS", f"{roas:.2f}" if not np.isnan(roas) else "N/A")
    c4.metric("Customer Count", f"{cust.shape[0]:,}" if not cust.empty else "N/A")

# Channel performance (horizontal bar)
def channel_performance():
    st.subheader("Channel Performance Comparison")
    df = df_or_warn('campaign')
    if df.empty:
        return
    metric = st.selectbox("Metric", ['revenue', 'conversions', 'roas'], index=0, key="chan_metric")
    if metric not in df.columns:
        st.warning(f"Metric '{metric}' not found.")
        return
    agg = df.groupby('channel', dropna=False)[metric].sum().reset_index().sort_values(metric)
    fig = px.bar(agg, x=metric, y='channel', orientation='h', text_auto=True, title=f"Total {metric.title()} by Channel")
    st.plotly_chart(fig, use_container_width=True)

# Revenue trend
def revenue_trend():
    st.subheader("Revenue Trend Over Time")
    df = df_or_warn('campaign')
    if df.empty or 'date' not in df.columns or 'revenue' not in df.columns:
        st.warning("campaign_performance.csv must contain 'date' and 'revenue'.")
        return
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="rt_dates")
    agg_level = st.selectbox("Aggregation level", ['Daily', 'Weekly', 'Monthly'], index=2, key="rt_agg")
    channels = st.multiselect("Channels", options=df['channel'].dropna().unique().tolist() if 'channel' in df.columns else [], default=df['channel'].dropna().unique().tolist() if 'channel' in df.columns else [], key="rt_channels")
    dff = df[(df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))]
    if channels:
        dff = dff[dff['channel'].isin(channels)]
    if agg_level == 'Daily':
        res = dff.groupby('date')['revenue'].sum().reset_index()
    elif agg_level == 'Weekly':
        res = dff.set_index('date').resample('W')['revenue'].sum().reset_index()
    else:
        res = dff.set_index('date').resample('M')['revenue'].sum().reset_index()
    fig = px.line(res, x='date', y='revenue', title=f"{agg_level} Revenue Trend")
    st.plotly_chart(fig, use_container_width=True)

# Cumulative conversions (stacked area)
def cumulative_conversions():
    st.subheader("Cumulative Conversions by Channel")
    df = df_or_warn('campaign')
    if df.empty or 'conversions' not in df.columns or 'date' not in df.columns:
        st.warning("campaign_performance.csv must include 'date' and 'conversions'.")
        return
    region = st.selectbox("Region", options=['All'] + (sorted(df['region'].dropna().unique().tolist()) if 'region' in df.columns else []), key="cum_region")
    dff = df.copy()
    if region != 'All':
        dff = dff[dff['region'] == region]
    tmp = dff.groupby(['date','channel'])['conversions'].sum().reset_index().sort_values('date')
    tmp['cum'] = tmp.groupby('channel')['conversions'].cumsum()
    fig = px.area(tmp, x='date', y='cum', color='channel', title='Cumulative Conversions')
    st.plotly_chart(fig, use_container_width=True)

# Histogram - age
def age_distribution():
    st.subheader("Customer Age Distribution")
    df = df_or_warn('customer')
    if df.empty or 'age' not in df.columns:
        st.warning("customer_data.csv missing 'age'.")
        return
    bins = st.slider("Bins", 5, 50, 20, key="age_bins")
    segs = ['All'] + (df['segment'].dropna().unique().tolist() if 'segment' in df.columns else [])
    seg = st.selectbox("Segment", options=segs, index=0, key="age_seg")
    dff = df.copy()
    if seg != 'All':
        dff = dff[dff['segment'] == seg]
    fig = px.histogram(dff, x='age', nbins=bins, title="Age Distribution")
    st.plotly_chart(fig, use_container_width=True)

# LTV box plot
def ltv_by_segment():
    st.subheader("Lifetime Value (LTV) by Segment")
    df = df_or_warn('customer')
    if df.empty or 'ltv' not in df.columns or 'segment' not in df.columns:
        st.warning("customer_data.csv must include 'ltv' and 'segment'.")
        return
    show_points = st.checkbox("Show individual points", key="ltv_points")
    fig = px.box(df, x='segment', y='ltv', points='all' if show_points else 'outliers', title='LTV by Segment')
    st.plotly_chart(fig, use_container_width=True)

# Violin plot - satisfaction
def satisfaction_violin():
    st.subheader("Satisfaction Score Distribution by NPS")
    df = df_or_warn('customer')
    if df.empty or 'satisfaction_score' not in df.columns or 'nps_category' not in df.columns:
        st.warning("customer_data.csv missing satisfaction_score or nps_category.")
        return
    split = 'acquisition_channel' if 'acquisition_channel' in df.columns else None
    if split:
        fig = px.violin(df, x='nps_category', y='satisfaction_score', color=split, box=True, points='outliers', title='Satisfaction by NPS and Channel')
    else:
        fig = px.violin(df, x='nps_category', y='satisfaction_score', box=True, points='outliers', title='Satisfaction by NPS')
    st.plotly_chart(fig, use_container_width=True)

# Income vs LTV scatter
def income_vs_ltv():
    st.subheader("Income vs LTV")
    df = df_or_warn('customer')
    if df.empty or 'income' not in df.columns or 'ltv' not in df.columns:
        st.warning("customer_data.csv must include 'income' and 'ltv'.")
        return
    show_trend = st.checkbox("Show trend line", key="income_trend")
    fig = px.scatter(df, x='income', y='ltv', color='segment' if 'segment' in df.columns else None, hover_data=['customer_id'] if 'customer_id' in df.columns else None, title="Income vs LTV")
    if show_trend:
        sub = df.dropna(subset=['income','ltv'])
        if len(sub) > 1:
            lr = LinearRegression()
            try:
                lr.fit(sub[['income']], sub['ltv'])
                xs = np.linspace(sub['income'].min(), sub['income'].max(), 100)
                ys = lr.predict(xs.reshape(-1,1))
                fig.add_scatter(x=xs, y=ys, mode='lines', name='Trendline', line=dict(color=PRIMARY))
            except Exception:
                pass
    st.plotly_chart(fig, use_container_width=True)

# Bubble chart - CTR vs Conversion Rate
def channel_bubble():
    st.subheader("Channel CTR vs Conversion Rate (bubble)")
    df = df_or_warn('campaign')
    if df.empty or not set(['ctr','conversion_rate','spend']).issubset(df.columns):
        st.warning("campaign_performance missing ctr/conversion_rate/spend.")
        return
    agg = df.groupby('channel').agg({'ctr':'mean','conversion_rate':'mean','spend':'sum'}).reset_index().dropna()
    fig = px.scatter(agg, x='ctr', y='conversion_rate', size='spend', color='channel', hover_data=['spend'], title="CTR vs Conversion Rate by Channel")
    st.plotly_chart(fig, use_container_width=True)

# Correlation heatmap
def correlation_heatmap():
    st.subheader("Correlation Heatmap")
    df = df_or_warn('corr')
    if df.empty:
        return
    try:
        fig = px.imshow(df.values, x=df.columns, y=df.index, color_continuous_scale='RdBu', zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("correlation_matrix.csv must be a square matrix with row/column labels.")

# Calendar heatmap
def calendar_heatmap():
    st.subheader("Calendar Heatmap")
    df = df_or_warn('campaign')
    if df.empty or 'date' not in df.columns:
        st.warning("campaign_performance needs 'date'.")
        return
    metric_options = [c for c in ['revenue','impressions'] if c in df.columns]
    if not metric_options:
        st.warning("No suitable metric for calendar heatmap.")
        return
    metric = st.selectbox("Metric", metric_options, key="cal_metric")
    d = df.groupby('date')[metric].sum().reset_index()
    d['dow'] = d['date'].dt.weekday
    d['week'] = d['date'].dt.isocalendar().week
    pivot = d.pivot_table(index='dow', columns='week', values=metric, aggfunc='sum')
    fig = px.imshow(pivot, labels=dict(x='Week', y='Day of Week', color=metric), title="Calendar Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# Donut (attribution)
def donut_attribution():
    st.subheader("Attribution Model Comparison")
    df = df_or_warn('attribution')
    if df.empty or 'channel' not in df.columns:
        st.warning("channel_attribution.csv missing or invalid.")
        return
    models = [c for c in df.columns if c != 'channel']
    model = st.selectbox("Attribution model", models, key="attr_model")
    fig = px.pie(df, names='channel', values=model, hole=0.5, title=f"Attribution - {model}")
    st.plotly_chart(fig, use_container_width=True)

# Treemap products
def treemap_products():
    st.subheader("Product Sales Treemap")
    df = df_or_warn('product')
    if df.empty:
        return
    path = [c for c in ['category','subcategory','product_name'] if c in df.columns]
    if not path:
        st.warning("product_sales.csv missing hierarchy columns.")
        return
    fig = px.treemap(df, path=path, values='sales', color='profit_margin' if 'profit_margin' in df.columns else None, title="Product Treemap")
    st.plotly_chart(fig, use_container_width=True)

# Funnel
def funnel_viz():
    st.subheader("Conversion Funnel")
    df = df_or_warn('funnel')
    if df.empty or not set(['stage','visitors']).issubset(df.columns):
        st.warning("funnel_data.csv requires 'stage' and 'visitors'.")
        return
    fig = px.funnel(df, x='visitors', y='stage', title="Funnel")
    st.plotly_chart(fig, use_container_width=True)

# Learning curve (handles your uploaded columns)
def learning_curve_plot():
    st.subheader("Learning Curve")
    df = df_or_warn('learning_curve')
    if df.empty:
        return
    # expected: train_size, train_score, val_score (we normalized names on load)
    required = {'train_size','train_score','val_score'}
    if not required.issubset(set(df.columns)):
        st.warning("learning_curve.csv must include train_size, train_score, val_score (or training_size/validation_score). Attempting to remap if possible.")
        remap = {}
        if 'training_size' in df.columns:
            remap['training_size'] = 'train_size'
        if 'validation_score' in df.columns:
            remap['validation_score'] = 'val_score'
        if remap:
            df = df.rename(columns=remap)
        if not required.issubset(set(df.columns)):
            st.error("After attempting mapping, required columns are still missing.")
            return
    show_conf = st.checkbox("Show confidence bands", value=True, key="lc_conf")
    fig = px.line(df, x='train_size', y=['train_score','val_score'], labels={'value':'Score','variable':'Dataset'}, title="Learning Curve")
    st.plotly_chart(fig, use_container_width=True)

# Feature importance
def feature_importance_plot():
    st.subheader("Feature Importance")
    df = df_or_warn('feature_importance')
    if df.empty or not set(['feature','importance']).issubset(df.columns):
        st.warning("feature_importance.csv must contain feature and importance.")
        return
    asc = st.checkbox("Sort ascending", value=False, key="fi_sort")
    dfp = df.sort_values('importance', ascending=asc)
    fig = px.bar(dfp, x='importance', y='feature', orientation='h', error_x='std' if 'std' in df.columns else None, title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

# Confusion matrix & ROC
def confusion_matrix_viz():
    st.subheader("Confusion Matrix")
    df = df_or_warn('lead')
    if df.empty or not set(['actual_converted','predicted_probability']).issubset(df.columns):
        st.warning("lead_scoring_results.csv needs actual_converted and predicted_probability.")
        return
    thr = st.slider("Threshold", 0.0, 1.0, 0.5, key="conf_thr")
    preds = (df['predicted_probability'] >= thr).astype(int)
    ct = pd.crosstab(df['actual_converted'], preds, rownames=['Actual'], colnames=['Predicted'])
    fig = px.imshow(ct.values, x=ct.columns.astype(str), y=ct.index.astype(str), text_auto=True, labels=dict(x='Predicted', y='Actual'), title=f"Confusion Matrix (thr={thr:.2f})")
    st.plotly_chart(fig, use_container_width=True)

def roc_viz():
    st.subheader("ROC Curve")
    df = df_or_warn('lead')
    if df.empty or not set(['actual_converted','predicted_probability']).issubset(df.columns):
        return
    fpr, tpr, thr = roc_curve(df['actual_converted'], df['predicted_probability'])
    roc_auc = auc(fpr,tpr)
    fig = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC={roc_auc:.2f})", labels={'x':'FPR','y':'TPR'})
    fig.add_scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray'), name='Random')
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"AUC = {roc_auc:.3f}")

# Geographic analysis (lat/lon bubble map or top-states bar fallback)
def geographic_analysis():
    st.header("Geographic Analysis")
    df = df_or_warn('geo')
    if df.empty:
        return
    candidates = [c for c in ['total_revenue','total_customers','market_penetration','yoy_growth'] if c in df.columns]
    if not candidates:
        st.warning("geographic_data.csv must include at least one metric: total_revenue/total_customers/market_penetration/yoy_growth.")
        return
    default_metric = 'total_revenue' if 'total_revenue' in candidates else candidates[0]
    metric = st.selectbox("Metric", candidates, index=candidates.index(default_metric), key="geo_metric")
    # lat/lon detection
    lat_col = 'latitude' if 'latitude' in df.columns else ('lat' if 'lat' in df.columns else None)
    lon_col = 'longitude' if 'longitude' in df.columns else ('lon' if 'lon' in df.columns else None)
    if lat_col and lon_col:
        st.info("Rendering bubble map using latitude/longitude.")
        size_col = 'store_count' if 'store_count' in df.columns else None
        color_col = 'customer_satisfaction' if 'customer_satisfaction' in df.columns else metric
        fig = px.scatter_geo(df, lat=lat_col, lon=lon_col, size=size_col, color=color_col, hover_name='state' if 'state' in df.columns else None, projection="natural earth", title=f"{metric} by Location")
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)
        return
    # fallback to bar by state
    if 'state' in df.columns:
        st.info("Latitude/longitude not present — showing top states by metric.")
        bar = df.groupby('state')[metric].sum().reset_index().sort_values(metric, ascending=False).head(20)
        fig = px.bar(bar, x=metric, y='state', orientation='h', title=f"Top States by {metric}")
        st.plotly_chart(fig, use_container_width=True)
        return
    st.warning("geographic_data.csv requires either latitude/longitude columns or a 'state' column.")

# ---------------------------
# PAGE ROUTER
# ---------------------------
st.sidebar.title("NovaMart Dashboard")
page = st.sidebar.radio("Navigate", [
    "Executive Overview",
    "Campaign Analytics",
    "Customer Insights",
    "Product Performance",
    "Geographic Analysis",
    "Attribution & Funnel",
    "ML Model Evaluation"
])

if page == "Executive Overview":
    st.markdown(f"<div style='font-size:28px;font-weight:700;color:{PRIMARY}'>Executive Overview</div>", unsafe_allow_html=True)
    kpi_overview()
    st.markdown("---")
    revenue_trend()
    st.markdown("---")
    channel_performance()

elif page == "Campaign Analytics":
    st.markdown(f"<div style='font-size:28px;font-weight:700;color:{PRIMARY}'>Campaign Analytics</div>", unsafe_allow_html=True)
    revenue_trend()
    st.markdown("---")
    cumulative_conversions()
    st.markdown("---")
    calendar_heatmap()
    st.markdown("---")
    channel_performance()

elif page == "Customer Insights":
    st.markdown(f"<div style='font-size:28px;font-weight:700;color:{PRIMARY}'>Customer Insights</div>", unsafe_allow_html=True)
    age_distribution()
    st.markdown("---")
    ltv_by_segment()
    st.markdown("---")
    satisfaction_violin()
    st.markdown("---")
    income_vs_ltv()
    st.markdown("---")
    channel_bubble()

elif page == "Product Performance":
    st.markdown(f"<div style='font-size:28px;font-weight:700;color:{PRIMARY}'>Product Performance</div>", unsafe_allow_html=True)
    treemap_products()

elif page == "Geographic Analysis":
    st.markdown(f"<div style='font-size:28px;font-weight:700;color:{PRIMARY}'>Geographic Analysis</div>", unsafe_allow_html=True)
    geographic_analysis()

elif page == "Attribution & Funnel":
    st.markdown(f"<div style='font-size:28px;font-weight:700;color:{PRIMARY}'>Attribution & Funnel</div>", unsafe_allow_html=True)
    donut_attribution()
    st.markdown("---")
    funnel_viz()
    st.markdown("---")
    correlation_heatmap()

elif page == "ML Model Evaluation":
    st.markdown(f"<div style='font-size:28px;font-weight:700;color:{PRIMARY}'>ML Model Evaluation</div>", unsafe_allow_html=True)
    confusion_matrix_viz()
    st.markdown("---")
    roc_viz()
    st.markdown("---")
    learning_curve_plot()
    st.markdown("---")
    feature_importance_plot()

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Built for: Streamlit Data Visualization Assignment")
st.sidebar.write("Author: Gagandeep Singh")
