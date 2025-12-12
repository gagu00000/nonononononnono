# app.py - NovaMart Marketing Analytics Dashboard (clean, robust, root-CSV)
# Place all 11 CSVs in the same folder as this file (root).
# Streamlit requirements: streamlit, pandas, numpy, plotly, scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LinearRegression

# ---------------------------
# Page config & Plotly theme
# ---------------------------
st.set_page_config(page_title="NovaMart Marketing Dashboard", layout="wide", initial_sidebar_state="expanded")

# Create a Plotly dark template (keeps traces visible)
pio.templates["nova_dark"] = pio.templates["plotly_dark"]
pio.templates["nova_dark"].layout.update({
    "paper_bgcolor": "#0B0F14",
    "plot_bgcolor": "#0B0F14",
    "font": {"color": "#FFFFFF"},
    "geo": {"bgcolor": "#0B0F14", "lakecolor": "#0B0F14", "landcolor": "#111213", "subunitcolor": "#444444"},
})
pio.templates.default = "nova_dark"

# ---------------------------
# Light-touch CSS (doesn't break Plotly)
# ---------------------------
st.markdown(
    """
    <style>
    /* App background + text */
    .stApp { background-color: #0B0F14 !important; color: #E6EEF6 !important; }
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0F1720 !important; color: #E6EEF6 !important; }
    /* Container cards */
    .css-1d391kg, .css-1r6slb0, .css-12w0qpk { background-color: transparent !important; }
    /* Metric readability */
    div[data-testid="metric-container"] .stMetricLabel, div[data-testid="metric-container"] .stMetricValue {
        color: #E6EEF6 !important;
    }
    /* Inputs */
    .stSelectbox, .stMultiSelect, .stSlider, .stDateInput, .stRadio, .stTextInput {
        color: #E6EEF6 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Data loading helpers
# ---------------------------
@st.cache_data
def safe_read_csv(path, parse_dates=None):
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_all_data():
    # Files expected at repo root
    files = {
        'campaign': "campaign_performance.csv",
        'channel_attribution': "channel_attribution.csv",
        'corr': "correlation_matrix.csv",
        'customer': "customer_data.csv",
        'journey': "customer_journey.csv",
        'feature_importance': "feature_importance.csv",
        'funnel': "funnel_data.csv",
        'geo': "geographic_data.csv",
        'lead': "lead_scoring_results.csv",
        'learning_curve': "learning_curve.csv",
        'product': "product_sales.csv"
    }
    data = {}
    for k, fname in files.items():
        if k == 'campaign':
            data[k] = safe_read_csv(fname, parse_dates=['date'])
        else:
            data[k] = safe_read_csv(fname)
    # Enrich campaign dates if available
    if not data['campaign'].empty and 'date' in data['campaign'].columns:
        try:
            data['campaign']['date'] = pd.to_datetime(data['campaign']['date'])
            data['campaign']['year'] = data['campaign']['date'].dt.year
            data['campaign']['month'] = data['campaign']['date'].dt.strftime('%B')
            data['campaign']['quarter'] = data['campaign']['date'].dt.to_period('Q').astype(str)
            data['campaign']['day_of_week'] = data['campaign']['date'].dt.day_name()
        except Exception:
            pass
    return data

data = load_all_data()

# ---------------------------
# utility
# ---------------------------
def df_or_warn(key):
    df = data.get(key)
    if df is None or df.empty:
        st.warning(f"Dataset `{key}` not found or empty. Upload `{key}.csv` to the app folder to enable related charts.")
        return pd.DataFrame()
    return df.copy()

def format_currency(v):
    try:
        return f"₹{v:,.0f}"
    except Exception:
        return v

# ---------------------------
# Visualizations
# ---------------------------
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
        c1.metric("Total Revenue", format_currency(total_rev))
        c2.metric("Total Conversions", f"{int(total_conv):,}")
        c3.metric("Total Spend", format_currency(total_spend))
        c4.metric("ROAS", f"{roas:.2f}" if not np.isnan(roas) else "N/A")
    c4.metric("Customer Count", f"{cust.shape[0]:,}" if not cust.empty else "N/A")

# Channel bar
def channel_bar():
    st.subheader("Channel Performance")
    df = df_or_warn('campaign')
    if df.empty:
        return
    metric = st.selectbox("Metric", ['revenue', 'conversions', 'roas'], index=0, key="chan_metric")
    if metric not in df.columns:
        st.warning(f"Metric '{metric}' not found in campaign data.")
        return
    agg = df.groupby('channel', dropna=False)[metric].sum().reset_index().sort_values(metric, ascending=True)
    fig = px.bar(agg, x=metric, y='channel', orientation='h', text_auto=True, title=f"Total {metric.title()} by Channel")
    st.plotly_chart(fig, use_container_width=True)

# Revenue trend
def revenue_trend():
    st.subheader("Revenue Trend")
    df = df_or_warn('campaign')
    if df.empty or 'date' not in df.columns or 'revenue' not in df.columns:
        st.warning("campaign_performance.csv must include 'date' and 'revenue' to show trends.")
        return
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="rev_dates")
    agg = st.selectbox("Aggregation", ['Daily','Weekly','Monthly'], index=2, key="rev_agg")
    channels = st.multiselect("Channels", options=df['channel'].dropna().unique().tolist() if 'channel' in df.columns else [], default=df['channel'].dropna().unique().tolist() if 'channel' in df.columns else [], key="rev_channels")
    dff = df[(df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))]
    if channels:
        dff = dff[dff['channel'].isin(channels)]
    if agg == 'Daily':
        res = dff.groupby('date')['revenue'].sum().reset_index()
    elif agg == 'Weekly':
        res = dff.set_index('date').resample('W')['revenue'].sum().reset_index()
    else:
        res = dff.set_index('date').resample('M')['revenue'].sum().reset_index()
    fig = px.line(res, x='date', y='revenue', title=f"{agg} Revenue Trend")
    st.plotly_chart(fig, use_container_width=True)

# Cumulative conversions area
def cumulative_area():
    st.subheader("Cumulative Conversions by Channel")
    df = df_or_warn('campaign')
    if df.empty or 'conversions' not in df.columns or 'date' not in df.columns:
        st.warning("campaign_performance requires 'date' and 'conversions'.")
        return
    region = st.selectbox("Region", options=['All'] + sorted(df['region'].dropna().unique().tolist()) if 'region' in df.columns else ['All'], key="cum_region")
    dff = df.copy()
    if region != 'All':
        dff = dff[dff['region'] == region]
    tmp = dff.groupby(['date','channel'])['conversions'].sum().reset_index().sort_values('date')
    tmp['cum'] = tmp.groupby('channel')['conversions'].cumsum()
    fig = px.area(tmp, x='date', y='cum', color='channel', title='Cumulative Conversions')
    st.plotly_chart(fig, use_container_width=True)

# Customer age histogram
def age_hist():
    st.subheader("Customer Age Distribution")
    df = df_or_warn('customer')
    if df.empty or 'age' not in df.columns:
        st.warning("customer_data.csv missing 'age' column.")
        return
    bins = st.slider("Bins", 5, 50, 20, key="age_bins")
    segs = ['All'] + (df['segment'].dropna().unique().tolist() if 'segment' in df.columns else [])
    seg = st.selectbox("Segment", options=segs, index=0, key="age_seg")
    dff = df.copy()
    if seg != 'All':
        dff = dff[dff['segment'] == seg]
    fig = px.histogram(dff, x='age', nbins=bins, title="Age Distribution")
    st.plotly_chart(fig, use_container_width=True)

# LTV boxplot
def ltv_box():
    st.subheader("LTV by Segment")
    df = df_or_warn('customer')
    if df.empty or 'ltv' not in df.columns or 'segment' not in df.columns:
        st.warning("customer_data.csv requires 'ltv' and 'segment'.")
        return
    show_points = st.checkbox("Show individual points", key="ltv_points")
    fig = px.box(df, x='segment', y='ltv', points='all' if show_points else 'outliers', title="LTV by Segment")
    st.plotly_chart(fig, use_container_width=True)

# Satisfaction violin
def satisfaction_violin():
    st.subheader("Satisfaction by NPS")
    df = df_or_warn('customer')
    if df.empty or 'satisfaction_score' not in df.columns or 'nps_category' not in df.columns:
        st.warning("customer_data.csv must include 'satisfaction_score' and 'nps_category'.")
        return
    split = st.selectbox("Split violin by", options=['None'] + (df['acquisition_channel'].dropna().unique().tolist() if 'acquisition_channel' in df.columns else []), key="violin_split")
    if split == 'None' or 'acquisition_channel' not in df.columns:
        fig = px.violin(df, x='nps_category', y='satisfaction_score', box=True, points='outliers', title="Satisfaction by NPS")
    else:
        fig = px.violin(df, x='nps_category', y='satisfaction_score', color='acquisition_channel', box=True, points='outliers', title="Satisfaction by NPS and Channel")
    st.plotly_chart(fig, use_container_width=True)

# Income vs LTV scatter
def income_ltv_scatter():
    st.subheader("Income vs LTV")
    df = df_or_warn('customer')
    if df.empty or 'income' not in df.columns or 'ltv' not in df.columns:
        st.warning("customer_data.csv needs 'income' and 'ltv'.")
        return
    show_trend = st.checkbox("Show trend line", key="trend_toggle")
    fig = px.scatter(df, x='income', y='ltv', color='segment' if 'segment' in df.columns else None, hover_data=['customer_id'] if 'customer_id' in df.columns else None, title="Income vs LTV")
    if show_trend:
        sub = df.dropna(subset=['income','ltv'])
        if len(sub) > 1:
            lr = LinearRegression()
            X = sub['income'].values.reshape(-1,1)
            y = sub['ltv'].values
            try:
                lr.fit(X,y)
                xs = np.linspace(sub['income'].min(), sub['income'].max(), 100)
                ys = lr.predict(xs.reshape(-1,1))
                fig.add_scatter(x=xs, y=ys, mode='lines', name='Trendline', line=dict(color='white'))
            except Exception:
                pass
    st.plotly_chart(fig, use_container_width=True)

# Channel bubble
def channel_bubble():
    st.subheader("CTR vs Conversion Rate by Channel")
    df = df_or_warn('campaign')
    if df.empty or not set(['ctr','conversion_rate','spend']).issubset(df.columns):
        st.warning("campaign_performance needs 'ctr','conversion_rate','spend' to show bubble chart.")
        return
    agg = df.groupby('channel').agg({'ctr':'mean','conversion_rate':'mean','spend':'sum'}).reset_index().dropna()
    fig = px.scatter(agg, x='ctr', y='conversion_rate', size='spend', color='channel', hover_data=['spend'], title="Channel Performance Matrix")
    st.plotly_chart(fig, use_container_width=True)

# Correlation heatmap
def corr_heatmap():
    st.subheader("Correlation Matrix")
    df = df_or_warn('corr')
    if df.empty:
        return
    try:
        fig = px.imshow(df.values, x=df.columns, y=df.index, color_continuous_scale='RdBu', zmin=-1, zmax=1, title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("Correlation matrix not in expected format (square).")

# Calendar heatmap
def calendar_heatmap():
    st.subheader("Calendar Heatmap (Daily revenue intensity)")
    df = df_or_warn('campaign')
    if df.empty or 'date' not in df.columns:
        st.warning("campaign_performance must include 'date' for calendar.")
        return
    metric = st.selectbox("Metric", options=['revenue','impressions'] if set(['revenue','impressions']).issubset(df.columns) else [c for c in df.columns if c in ['revenue','impressions']], key="cal_metric")
    if metric not in df.columns:
        st.warning("Selected metric not in campaign data.")
        return
    df2 = df.groupby('date')[metric].sum().reset_index()
    df2['dow'] = df2['date'].dt.weekday
    df2['week'] = df2['date'].dt.isocalendar().week
    pivot = df2.pivot_table(index='dow', columns='week', values=metric, aggfunc='sum')
    fig = px.imshow(pivot, labels=dict(x='Week', y='Day of Week', color=metric), title="Calendar Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# Donut attribution
def donut_attribution():
    st.subheader("Attribution Model Comparison")
    df = df_or_warn('channel_attribution')
    if df.empty or 'channel' not in df.columns:
        st.warning("channel_attribution.csv missing or invalid.")
        return
    models = [c for c in df.columns if c != 'channel']
    model = st.selectbox("Attribution model", options=models, index=0, key="attr_model")
    fig = px.pie(df, names='channel', values=model, hole=0.5, title=f"Attribution - {model}")
    st.plotly_chart(fig, use_container_width=True)

# Treemap products
def treemap_products():
    st.subheader("Product Sales Treemap")
    df = df_or_warn('product')
    if df.empty or 'sales' not in df.columns:
        st.warning("product_sales.csv missing or doesn't have 'sales'.")
        return
    path = [c for c in ['category','subcategory','product_name'] if c in df.columns]
    fig = px.treemap(df, path=path, values='sales', color='profit_margin' if 'profit_margin' in df.columns else None, title="Product Hierarchy (Treemap)")
    st.plotly_chart(fig, use_container_width=True)

# Funnel chart
def funnel_chart():
    st.subheader("Conversion Funnel")
    df = df_or_warn('funnel')
    if df.empty or not set(['stage','visitors']).issubset(df.columns):
        st.warning("funnel_data.csv must have 'stage' and 'visitors'.")
        return
    fig = px.funnel(df, x='visitors', y='stage', title="Funnel")
    st.plotly_chart(fig, use_container_width=True)

# Model evaluation: confusion, ROC, learning, feature importance
def confusion_matrix():
    st.subheader("Confusion Matrix (Lead Scoring)")
    df = df_or_warn('lead')
    if df.empty or not set(['actual_converted','predicted_probability']).issubset(df.columns):
        st.warning("lead_scoring_results.csv needs 'actual_converted' and 'predicted_probability'.")
        return
    thr = st.slider("Threshold", 0.0, 1.0, 0.5, key="conf_thr")
    preds = (df['predicted_probability'] >= thr).astype(int)
    ct = pd.crosstab(df['actual_converted'], preds, rownames=['Actual'], colnames=['Predicted'])
    fig = px.imshow(ct.values, x=ct.columns.astype(str), y=ct.index.astype(str), text_auto=True, labels=dict(x='Predicted', y='Actual'), title=f"Confusion Matrix (thr={thr:.2f})")
    st.plotly_chart(fig, use_container_width=True)

def roc_plot():
    st.subheader("ROC Curve")
    df = df_or_warn('lead')
    if df.empty or not set(['actual_converted','predicted_probability']).issubset(df.columns):
        return
    fpr, tpr, thr = roc_curve(df['actual_converted'], df['predicted_probability'])
    roc_auc = auc(fpr, tpr)
    fig = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC={roc_auc:.2f})", labels={'x':'FPR','y':'TPR'})
    fig.add_scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='white'), name='Random')
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"AUC = {roc_auc:.3f}")

def learning_curve_plot():
    st.subheader("Learning Curve")
    df = df_or_warn('learning_curve')
    if df.empty or not set(['train_size','train_score','val_score']).issubset(df.columns):
        st.warning("learning_curve.csv must have train_size, train_score, val_score.")
        return
    fig = px.line(df, x='train_size', y=['train_score','val_score'], labels={'value':'Score','variable':'Dataset'}, title="Learning Curve")
    st.plotly_chart(fig, use_container_width=True)

def feature_importance_plot():
    st.subheader("Feature Importance")
    df = df_or_warn('feature_importance')
    if df.empty or not set(['feature','importance']).issubset(df.columns):
        st.warning("feature_importance.csv must have feature and importance.")
        return
    df_sorted = df.sort_values('importance', ascending=True)
    fig = px.bar(df_sorted, x='importance', y='feature', orientation='h', error_x='std' if 'std' in df.columns else None, title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Geographic analysis (robust)
# ---------------------------
def geographic_analysis():
    st.header("Geographic Analysis")
    df = df_or_warn('geo')
    if df.empty:
        return

    # Candidate metrics for mapping (prefer total_revenue)
    candidates = [c for c in ['total_revenue','total_customers','market_penetration','yoy_growth'] if c in df.columns]
    if not candidates:
        st.warning("geographic_data.csv needs at least one metric (total_revenue/total_customers/market_penetration/yoy_growth).")
        return
    default_metric = 'total_revenue' if 'total_revenue' in candidates else candidates[0]
    metric = st.selectbox("Metric", options=candidates, index=candidates.index(default_metric), key="geo_metric")

    # If lat/lon available -> bubble map (reliable)
    lat_col = 'latitude' if 'latitude' in df.columns else ('lat' if 'lat' in df.columns else None)
    lon_col = 'longitude' if 'longitude' in df.columns else ('lon' if 'lon' in df.columns else None)
    if lat_col and lon_col:
        st.info("Rendering bubble map using latitude/longitude.")
        size_col = 'store_count' if 'store_count' in df.columns else None
        color_col = 'customer_satisfaction' if 'customer_satisfaction' in df.columns else metric
        fig = px.scatter_geo(df, lat=lat_col, lon=lon_col, size=size_col, color=color_col,
                             hover_name='state' if 'state' in df.columns else None,
                             projection="natural earth",
                             title=f"{metric} across states (bubble map)")
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)
        return

    # If lat/lon not available but state is -> bar chart fallback
    if 'state' in df.columns:
        st.info("Latitude/longitude not found; showing top states by metric.")
        bar = df.groupby('state')[metric].sum().reset_index().sort_values(metric, ascending=False).head(20)
        fig = px.bar(bar, x=metric, y='state', orientation='h', title=f"Top states by {metric}")
        st.plotly_chart(fig, use_container_width=True)
        return

    st.warning("geographic_data.csv must contain either latitude/longitude columns or a 'state' column for mapping.")

# ---------------------------
# App router / pages
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
    kpi_overview()
    st.markdown("---")
    revenue_trend()
    st.markdown("---")
    channel_bar()

elif page == "Campaign Analytics":
    st.header("Campaign Analytics")
    revenue_trend()
    st.markdown("---")
    cumulative_area()
    st.markdown("---")
    calendar_heatmap()
    st.markdown("---")
    channel_bar()

elif page == "Customer Insights":
    st.header("Customer Insights")
    age_hist()
    st.markdown("---")
    ltv_box()
    st.markdown("---")
    satisfaction_violin()
    st.markdown("---")
    income_ltv_scatter()
    st.markdown("---")
    channel_bubble()

elif page == "Product Performance":
    st.header("Product Performance")
    treemap_products()

elif page == "Geographic Analysis":
    geographic_analysis()

elif page == "Attribution & Funnel":
    st.header("Attribution & Funnel")
    donut_attribution()
    st.markdown("---")
    funnel_chart()
    st.markdown("---")
    corr_heatmap()

elif page == "ML Model Evaluation":
    st.header("ML Model Evaluation")
    confusion_matrix()
    st.markdown("---")
    roc_plot()
    st.markdown("---")
    learning_curve_plot()
    st.markdown("---")
    feature_importance_plot()

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Built for: Masters of AI in Business — NovaMart")
st.sidebar.write("Author: Data Analyst")
