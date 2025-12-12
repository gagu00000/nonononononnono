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
# Geographic Analysis for India - Enhanced with Choropleth
# Replace the geographic_analysis() function in your app.py with this

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def geographic_analysis_india(data_dict):
    """Geographic Analysis focused on India with state choropleth map"""
    st.header("Geographic Analysis - India")
    
    df = df_or_warn(data_dict, 'geo')
    if df.empty:
        return
    
    # Available metrics
    candidates = [c for c in ['total_revenue', 'total_customers', 'market_penetration', 'yoy_growth'] 
                  if c in df.columns]
    
    if not candidates:
        st.warning("geographic_data.csv must include at least one metric.")
        return
    
    # Metric selector
    default_metric = 'total_revenue' if 'total_revenue' in candidates else candidates[0]
    metric = st.selectbox("Select Metric", candidates, 
                          index=candidates.index(default_metric), key="geo_metric")
    
    # Check for state column
    if 'state' not in df.columns:
        st.error("geographic_data.csv must include a 'state' column.")
        return
    
    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["🗺️ State Map", "📊 Top States", "📈 Trends"])
    
    with tab1:
        render_india_state_map(df, metric)
    
    with tab2:
        render_state_rankings(df, metric)
    
    with tab3:
        render_state_trends(df, metric)


def render_india_state_map(df, metric):
    """Render choropleth map of Indian states"""
    
    # State code mapping for India (ISO 3166-2:IN)
    state_codes = {
        'andhra pradesh': 'AP', 'arunachal pradesh': 'AR', 'assam': 'AS',
        'bihar': 'BR', 'chhattisgarh': 'CG', 'goa': 'GA', 'gujarat': 'GJ',
        'haryana': 'HR', 'himachal pradesh': 'HP', 'jharkhand': 'JH',
        'karnataka': 'KA', 'kerala': 'KL', 'madhya pradesh': 'MP',
        'maharashtra': 'MH', 'manipur': 'MN', 'meghalaya': 'ML',
        'mizoram': 'MZ', 'nagaland': 'NL', 'odisha': 'OR', 'punjab': 'PB',
        'rajasthan': 'RJ', 'sikkim': 'SK', 'tamil nadu': 'TN',
        'telangana': 'TG', 'tripura': 'TR', 'uttar pradesh': 'UP',
        'uttarakhand': 'UK', 'west bengal': 'WB',
        'andaman and nicobar': 'AN', 'chandigarh': 'CH',
        'dadra and nagar haveli': 'DN', 'daman and diu': 'DD',
        'delhi': 'DL', 'jammu and kashmir': 'JK', 'ladakh': 'LA',
        'lakshadweep': 'LD', 'puducherry': 'PY'
    }
    
    # Prepare data
    df_plot = df.copy()
    df_plot['state_lower'] = df_plot['state'].str.lower().str.strip()
    df_plot['state_code'] = df_plot['state_lower'].map(state_codes)
    
    # Aggregate by state
    state_agg = df_plot.groupby(['state', 'state_code']).agg({
        metric: 'sum'
    }).reset_index()
    
    state_agg = state_agg.dropna(subset=['state_code'])
    
    if state_agg.empty:
        st.warning("No valid state data found. Please check state names in your CSV.")
        return
    
    # Create India choropleth using scatter_geo with India focus
    fig = px.scatter_geo(
        state_agg,
        locations='state_code',
        locationmode='ISO-3',
        color=metric,
        hover_name='state',
        size=metric,
        color_continuous_scale='Blues',
        title=f"{metric.replace('_', ' ').title()} Across Indian States",
        size_max=50
    )
    
    # Configure geography for India
    fig.update_geos(
        scope='asia',
        center=dict(lat=22.5, lon=78.9),  # Center of India
        projection_scale=4.5,  # Zoom level
        visible=True,
        resolution=50,
        showcountries=True,
        countrycolor='lightgray',
        showcoastlines=True,
        coastlinecolor='white',
        showland=True,
        landcolor='rgb(250, 250, 250)',
        bgcolor='rgb(255, 255, 255)',
        lataxis_range=[6, 37],
        lonaxis_range=[68, 98]
    )
    
    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(
            title=metric.replace('_', ' ').title(),
            thickness=15,
            len=0.7
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional breakdown
    st.subheader("🌏 Regional Breakdown")
    
    # Define regions (you can customize this based on your business regions)
    region_mapping = {
        'north': ['punjab', 'haryana', 'himachal pradesh', 'jammu and kashmir', 
                  'ladakh', 'delhi', 'chandigarh', 'uttarakhand', 'uttar pradesh'],
        'south': ['karnataka', 'kerala', 'tamil nadu', 'andhra pradesh', 
                  'telangana', 'puducherry'],
        'east': ['west bengal', 'odisha', 'bihar', 'jharkhand', 'assam', 
                 'arunachal pradesh', 'manipur', 'meghalaya', 'mizoram', 
                 'nagaland', 'sikkim', 'tripura'],
        'west': ['maharashtra', 'gujarat', 'goa', 'rajasthan', 'madhya pradesh', 
                 'chhattisgarh']
    }
    
    df_plot['region'] = df_plot['state_lower'].apply(
        lambda x: next((k for k, v in region_mapping.items() if x in v), 'other')
    )
    
    region_agg = df_plot.groupby('region')[metric].sum().reset_index()
    region_agg = region_agg[region_agg['region'] != 'other'].sort_values(metric, ascending=False)
    
    if not region_agg.empty:
        fig_region = px.bar(
            region_agg, 
            x='region', 
            y=metric,
            text_auto='.2s',
            color='region',
            title=f"{metric.replace('_', ' ').title()} by Region"
        )
        fig_region.update_traces(textposition='outside')
        fig_region.update_layout(showlegend=False, xaxis_title="Region", 
                                 yaxis_title=metric.replace('_', ' ').title())
        st.plotly_chart(fig_region, use_container_width=True)


def render_state_rankings(df, metric):
    """Show top and bottom performing states"""
    
    state_agg = df.groupby('state')[metric].sum().reset_index().sort_values(metric, ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 States")
        top10 = state_agg.head(10)
        
        fig_top = px.bar(
            top10, 
            y='state', 
            x=metric,
            orientation='h',
            text_auto='.2s',
            color=metric,
            color_continuous_scale='Greens'
        )
        fig_top.update_traces(textposition='outside')
        fig_top.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        st.subheader("📉 Bottom 10 States")
        bottom10 = state_agg.tail(10)
        
        fig_bottom = px.bar(
            bottom10, 
            y='state', 
            x=metric,
            orientation='h',
            text_auto='.2s',
            color=metric,
            color_continuous_scale='Reds'
        )
        fig_bottom.update_traces(textposition='outside')
        fig_bottom.update_layout(showlegend=False, yaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig_bottom, use_container_width=True)
    
    # Summary metrics
    st.subheader("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total States", len(state_agg))
    with col2:
        st.metric(f"Total {metric.replace('_', ' ').title()}", f"{state_agg[metric].sum():,.0f}")
    with col3:
        st.metric("Average per State", f"{state_agg[metric].mean():,.0f}")
    with col4:
        st.metric("Median per State", f"{state_agg[metric].median():,.0f}")


def render_state_trends(df, metric):
    """Show trends if data includes additional dimensions"""
    st.subheader("📈 State Comparison")
    
    # Multi-select for states
    all_states = sorted(df['state'].dropna().unique().tolist())
    
    if len(all_states) == 0:
        st.info("No state data available for trends.")
        return
    
    default_states = all_states[:5] if len(all_states) >= 5 else all_states
    selected_states = st.multiselect(
        "Select states to compare",
        options=all_states,
        default=default_states,
        key="state_compare"
    )
    
    if not selected_states:
        st.info("Please select at least one state to view comparison.")
        return
    
    # Filter and compare
    df_filtered = df[df['state'].isin(selected_states)]
    
    # If there are multiple metrics, show comparison
    numeric_cols = df_filtered.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) > 1:
        comparison = df_filtered.groupby('state')[numeric_cols].sum().reset_index()
        
        fig = px.bar(
            comparison.melt(id_vars='state', value_vars=numeric_cols),
            x='state',
            y='value',
            color='variable',
            barmode='group',
            title="Multi-Metric Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Single metric comparison
        comparison = df_filtered.groupby('state')[metric].sum().reset_index()
        fig = px.bar(comparison, x='state', y=metric, text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("📋 Detailed Data")
    st.dataframe(
        df_filtered.groupby('state')[numeric_cols].sum().reset_index(),
        use_container_width=True,
        hide_index=True
    )


def df_or_warn(data_dict, key):
    """Get dataframe from data dict with warning if missing"""
    df = data_dict.get(key)
    if df is None or df.empty:
        st.warning(f"Dataset `{key}` missing or empty.")
        return pd.DataFrame()
    return df.copy()


# INSTRUCTIONS FOR INTEGRATION:
# 1. Replace your geographic_analysis() function with geographic_analysis_india()
# 2. Update the function call in your page router:
#    elif page == "Geographic Analysis":
#        geographic_analysis_india(data)

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
