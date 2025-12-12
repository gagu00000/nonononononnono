# app.py - NovaMart Marketing Analytics Dashboard (FIXED VERSION)
# Critical fixes applied:
# 1. Removed global 'data' variable - now passed as parameter
# 2. Fixed date range handling
# 3. Removed redundant learning curve remapping
# 4. Added proper error messages
# 5. Fixed geographic column normalization

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc
from datetime import datetime

# ---------------------------
# THEME SETTINGS
# ---------------------------
APP_BG = "#0E1117"
TEXT_COLOR = "#FFFFFF"
SIDEBAR_BLUE = "#1F4E79"
PRIMARY = "#0B3D91"
ACCENT = "#2B8CC4"
PALETTE = [PRIMARY, ACCENT, "#66A3D2", "#B2D4EE", "#F4B400"]

st.set_page_config(page_title="NovaMart Marketing Dashboard", layout="wide", initial_sidebar_state="expanded")

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

# CSS
st.markdown(f"""
<style>
body, .stApp, .block-container {{
  background-color: {APP_BG} !important;
  color: {TEXT_COLOR} !important;
}}
section[data-testid="stSidebar"] {{
  background-color: {SIDEBAR_BLUE} !important;
  color: {TEXT_COLOR} !important;
}}
div[data-testid="metric-container"] {{
  background: rgba(255,255,255,0.03) !important;
  padding: 10px !important;
  border-radius: 8px;
}}
div[data-testid="metric-container"] .stMetricLabel, div[data-testid="metric-container"] .stMetricValue {{
  color: {TEXT_COLOR} !important;
}}
.stSelectbox, .stTextInput, .stNumberInput, .stDateInput, .stMultiSelect, .stSlider {{
  color: {TEXT_COLOR} !important;
}}
h1, h2, h3, h4, h5, p, span, label {{
  color: {TEXT_COLOR} !important;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# DATA LOADING
# ---------------------------
@st.cache_data
def safe_read_csv(path, parse_dates=None):
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading {path}: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_all():
    """Load and normalize all CSV files"""
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
    
    # Load campaign with date parsing
    data['campaign'] = safe_read_csv(files['campaign'], parse_dates=['date'])
    
    # Load other files
    for k in ['customer', 'product', 'lead', 'feature_importance', 
              'learning_curve', 'geo', 'attribution', 'funnel', 'journey', 'corr']:
        data[k] = safe_read_csv(files[k])

    # Enrich campaign
    if not data['campaign'].empty and 'date' in data['campaign'].columns:
        try:
            df = data['campaign'].copy()
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.strftime('%B')
            df['quarter'] = df['date'].dt.to_period('Q').astype(str)
            data['campaign'] = df
        except Exception as e:
            st.warning(f"Could not enrich campaign dates: {e}")

    # Normalize learning_curve columns ONCE HERE
    if not data['learning_curve'].empty:
        lc = data['learning_curve'].copy()
        rename_map = {}
        if 'training_size' in lc.columns and 'train_size' not in lc.columns:
            rename_map['training_size'] = 'train_size'
        if 'validation_score' in lc.columns and 'val_score' not in lc.columns:
            rename_map['validation_score'] = 'val_score'
        if rename_map:
            data['learning_curve'] = lc.rename(columns=rename_map)

    # Normalize geographic columns
    if not data['geo'].empty:
        geo = data['geo'].copy()
        if 'lat' in geo.columns and 'latitude' not in geo.columns:
            geo = geo.rename(columns={'lat': 'latitude'})
        if 'lon' in geo.columns and 'longitude' not in geo.columns:
            geo = geo.rename(columns={'lon': 'longitude'})
        data['geo'] = geo

    return data

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def df_or_warn(data_dict, key):
    """Get dataframe from data dict with warning if missing"""
    df = data_dict.get(key)
    if df is None or df.empty:
        st.warning(f"Dataset `{key}` missing or empty. Upload the CSV to enable related charts.")
        return pd.DataFrame()
    return df.copy()

def money(x):
    """Format as Indian Rupees"""
    if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)):
        return f"₹{x:,.0f}"
    return "N/A"

def safe_date_range(date_input):
    """Handle date_input which can be single date or tuple"""
    if isinstance(date_input, tuple) and len(date_input) == 2:
        dates = [pd.to_datetime(d) for d in date_input]
        return tuple(sorted(dates))
    elif isinstance(date_input, (datetime, pd.Timestamp)):
        # Single date selected - use same date for both
        dt = pd.to_datetime(date_input)
        return (dt, dt)
    else:
        # Fallback
        return (pd.Timestamp.now(), pd.Timestamp.now())

# ---------------------------
# VISUALIZATION FUNCTIONS (FIXED)
# ---------------------------

def kpi_overview(data_dict):
    st.header("Executive Overview")
    df = df_or_warn(data_dict, 'campaign')
    cust = df_or_warn(data_dict, 'customer')
    
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
        roas = total_rev / total_spend if total_spend > 0 else np.nan
        
        c1.metric("Total Revenue", money(total_rev))
        c2.metric("Total Conversions", f"{int(total_conv):,}")
        c3.metric("Total Spend", money(total_spend))
        c4.metric("ROAS", f"{roas:.2f}" if not np.isnan(roas) else "N/A")
    
    c4.metric("Customer Count", f"{cust.shape[0]:,}" if not cust.empty else "N/A")

def channel_performance(data_dict):
    st.subheader("Channel Performance Comparison")
    df = df_or_warn(data_dict, 'campaign')
    if df.empty:
        return
    
    metric = st.selectbox("Metric", ['revenue', 'conversions', 'roas'], index=0, key="chan_metric")
    if metric not in df.columns:
        st.warning(f"Metric '{metric}' not found.")
        return
    
    agg = df.groupby('channel', dropna=False)[metric].sum().reset_index().sort_values(metric)
    fig = px.bar(agg, x=metric, y='channel', orientation='h', text_auto=True, 
                 title=f"Total {metric.title()} by Channel")
    st.plotly_chart(fig, use_container_width=True)

def revenue_trend(data_dict):
    st.subheader("Revenue Trend Over Time")
    df = df_or_warn(data_dict, 'campaign')
    
    if df.empty or 'date' not in df.columns or 'revenue' not in df.columns:
        st.warning("campaign_performance.csv must contain 'date' and 'revenue'.")
        return
    
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    date_input = st.date_input(
        "Date range", 
        value=(min_date, max_date), 
        min_value=min_date, 
        max_value=max_date, 
        key="rt_dates"
    )
    
    # FIXED: Handle date range properly
    start_date, end_date = safe_date_range(date_input)
    
    agg_level = st.selectbox("Aggregation level", ['Daily', 'Weekly', 'Monthly'], index=2, key="rt_agg")
    
    channels = st.multiselect(
        "Channels", 
        options=df['channel'].dropna().unique().tolist() if 'channel' in df.columns else [], 
        default=df['channel'].dropna().unique().tolist() if 'channel' in df.columns else [], 
        key="rt_channels"
    )
    
    # Filter data
    dff = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    if channels:
        dff = dff[dff['channel'].isin(channels)]
    
    if dff.empty:
        st.info("No data for selected filters")
        return
    
    # Aggregate
    if agg_level == 'Daily':
        res = dff.groupby('date')['revenue'].sum().reset_index()
    elif agg_level == 'Weekly':
        res = dff.set_index('date').resample('W')['revenue'].sum().reset_index()
    else:
        res = dff.set_index('date').resample('M')['revenue'].sum().reset_index()
    
    fig = px.line(res, x='date', y='revenue', title=f"{agg_level} Revenue Trend")
    st.plotly_chart(fig, use_container_width=True)

def learning_curve_plot(data_dict):
    st.subheader("Learning Curve")
    df = df_or_warn(data_dict, 'learning_curve')
    
    if df.empty:
        return
    
    # FIXED: Removed redundant remapping - already done in load_all()
    required = {'train_size', 'train_score', 'val_score'}
    if not required.issubset(set(df.columns)):
        st.error("learning_curve.csv must include train_size, train_score, val_score (or we'll auto-remap training_size/validation_score)")
        return
    
    show_conf = st.checkbox("Show confidence bands", value=True, key="lc_conf")
    
    fig = px.line(df, x='train_size', y=['train_score', 'val_score'], 
                  labels={'value': 'Score', 'variable': 'Dataset'}, 
                  title="Learning Curve")
    st.plotly_chart(fig, use_container_width=True)

def income_vs_ltv(data_dict):
    st.subheader("Income vs LTV")
    df = df_or_warn(data_dict, 'customer')
    
    if df.empty or 'income' not in df.columns or 'ltv' not in df.columns:
        st.warning("customer_data.csv must include 'income' and 'ltv'.")
        return
    
    show_trend = st.checkbox("Show trend line", key="income_trend")
    
    fig = px.scatter(
        df, x='income', y='ltv', 
        color='segment' if 'segment' in df.columns else None, 
        hover_data=['customer_id'] if 'customer_id' in df.columns else None, 
        title="Income vs LTV"
    )
    
    if show_trend:
        sub = df.dropna(subset=['income', 'ltv'])
        if len(sub) > 1:
            try:
                lr = LinearRegression()
                lr.fit(sub[['income']], sub['ltv'])
                xs = np.linspace(sub['income'].min(), sub['income'].max(), 100)
                ys = lr.predict(xs.reshape(-1, 1))
                fig.add_scatter(x=xs, y=ys, mode='lines', name='Trendline', 
                               line=dict(color=PRIMARY))
            except Exception as e:
                st.warning(f"Could not compute trend line: {e}")
    
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# MAIN APP (FIXED)
# ---------------------------

# Load data ONCE and pass to functions
data = load_all()

st.sidebar.title("NovaMart Dashboard")

# Add refresh button
if st.sidebar.button("🔄 Reload Data"):
    st.cache_data.clear()
    st.rerun()

page = st.sidebar.radio("Navigate", [
    "Executive Overview",
    "Campaign Analytics",
    "Customer Insights",
])

if page == "Executive Overview":
    st.markdown(f"<div style='font-size:28px;font-weight:700;color:{PRIMARY}'>Executive Overview</div>", 
                unsafe_allow_html=True)
    kpi_overview(data)
    st.markdown("---")
    revenue_trend(data)
    st.markdown("---")
    channel_performance(data)

elif page == "Campaign Analytics":
    st.markdown(f"<div style='font-size:28px;font-weight:700;color:{PRIMARY}'>Campaign Analytics</div>", 
                unsafe_allow_html=True)
    revenue_trend(data)
    st.markdown("---")
    channel_performance(data)

elif page == "Customer Insights":
    st.markdown(f"<div style='font-size:28px;font-weight:700;color:{PRIMARY}'>Customer Insights</div>", 
                unsafe_allow_html=True)
    income_vs_ltv(data)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Built for: Masters of AI in Business — NovaMart")
st.sidebar.write("Author: Data Analyst")
