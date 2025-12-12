"""
app.py - NovaMart Marketing Analytics Dashboard
Full Streamlit app implementing the 20+ visualizations requested.

Usage:
- Place all 11 CSVs in the same folder as app.py (root)
- Install requirements: pip install -r requirements.txt
- Run locally: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LinearRegression

# ---------------------------
# Styling / Theme variables
# ---------------------------
PRIMARY = "#0B3D91"   # deep blue
ACCENT = "#F4B400"    # gold
BG = "#F5F7FA"

st.set_page_config(page_title="NovaMart Marketing Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG}; }}
    .big-title {{ font-size:34px; font-weight:700; color:{PRIMARY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Data loading with caching
# ---------------------------
DATA_PATH = "."  # CSVs are in the same folder as app.py

@st.cache_data
def safe_read_csv(path, parse_dates=None):
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_all():
    files = {
        'campaign': f"{DATA_PATH}/campaign_performance.csv",
        'customer': f"{DATA_PATH}/customer_data.csv",
        'product': f"{DATA_PATH}/product_sales.csv",
        'lead': f"{DATA_PATH}/lead_scoring_results.csv",
        'feature_importance': f"{DATA_PATH}/feature_importance.csv",
        'learning_curve': f"{DATA_PATH}/learning_curve.csv",
        'geo': f"{DATA_PATH}/geographic_data.csv",
        'attribution': f"{DATA_PATH}/channel_attribution.csv",
        'funnel': f"{DATA_PATH}/funnel_data.csv",
        'journey': f"{DATA_PATH}/customer_journey.csv",
        'corr': f"{DATA_PATH}/correlation_matrix.csv"
    }
    data = {}
    for k, p in files.items():
        if k == 'campaign':
            data[k] = safe_read_csv(p, parse_dates=['date'])
        else:
            data[k] = safe_read_csv(p)
    # Ensure 'date' in campaign is datetime and enrich
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

data = load_all()

# ---------------------------
# Utility helpers
# ---------------------------
def warn_missing(name):
    st.warning(f"Dataset `{name}` not found or empty. Upload `{name}.csv` to the app folder to enable related charts.")

def safe_df(name):
    df = data.get(name)
    if df is None or df.empty:
        warn_missing(name)
        return pd.DataFrame()
    return df.copy()

def currency_fmt(x):
    try:
        return f"₹{x:,.0f}"
    except Exception:
        return x

# ---------------------------
# Visualization functions
# ---------------------------

# Executive KPI cards
def kpi_cards():
    df = safe_df('campaign')
    cust = safe_df('customer')
    col1, col2, col3, col4 = st.columns(4)
    if df.empty:
        col1.metric("Total Revenue", "N/A")
        col2.metric("Total Conversions", "N/A")
        col3.metric("ROAS", "N/A")
        col4.metric("Customer Count", "N/A")
    else:
        total_rev = df['revenue'].sum()
        total_conv = df['conversions'].sum()
        total_spend = df['spend'].sum() if 'spend' in df.columns else 0
        roas = total_rev / total_spend if total_spend else np.nan
        col1.metric("Total Revenue", currency_fmt(total_rev))
        col2.metric("Total Conversions", f"{int(total_conv):,}")
        col3.metric("Total Spend", currency_fmt(total_spend))
        col4.metric("ROAS", f"{roas:.2f}" if not np.isnan(roas) else "N/A")
    col4.metric("Customer Count", f"{cust.shape[0]:,}" if not cust.empty else "N/A")

# Channel Performance horizontal bar
def channel_performance():
    df = safe_df('campaign')
    if df.empty:
        return
    metric = st.selectbox("Metric", ['revenue', 'conversions', 'roas'], index=0)
    agg = df.groupby('channel', dropna=False)[metric].sum().reset_index().sort_values(metric)
    fig = px.bar(agg, x=metric, y='channel', orientation='h', text=metric, title=f"Total {metric.title()} by Channel",
                 color_discrete_sequence=[PRIMARY])
    fig.update_layout(showlegend=False, height=420)
    st.plotly_chart(fig, use_container_width=True)

# Grouped bar - regional by quarter
def regional_by_quarter():
    df = safe_df('campaign')
    if df.empty:
        return
    if 'year' not in df.columns:
        st.warning("campaign_performance must include a date column to compute year/quarter.")
        return
    years = sorted(df['year'].dropna().unique().tolist())
    if not years:
        st.warning("No year data available.")
        return
    year = st.selectbox("Select Year", options=years, index=len(years)-1)
    tmp = df[df['year']==year].groupby(['region', 'quarter'])['revenue'].sum().reset_index()
    fig = px.bar(tmp, x='region', y='revenue', color='quarter', barmode='group',
                 title=f"Revenue by Region - {year}", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)

# Stacked spend
def stacked_campaign_spend():
    df = safe_df('campaign')
    if df.empty:
        return
    mode = st.radio("View", ['Absolute', '100% (Normalized)'])
    tmp = df.groupby(['month', 'campaign_type'], sort=False)['spend'].sum().reset_index()
    # ensure month order
    month_order = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    if 'month' in tmp.columns:
        tmp['month'] = pd.Categorical(tmp['month'], categories=month_order, ordered=True)
        tmp = tmp.sort_values('month')
    if mode == '100% (Normalized)':
        tmp['spend_pct'] = tmp.groupby('month')['spend'].apply(lambda x: x / x.sum())
        fig = px.bar(tmp, x='month', y='spend_pct', color='campaign_type', title='Monthly Spend Composition (100%)')
    else:
        fig = px.bar(tmp, x='month', y='spend', color='campaign_type', title='Monthly Spend by Campaign Type')
    st.plotly_chart(fig, use_container_width=True)

# Revenue trend time series
def revenue_trend():
    df = safe_df('campaign')
    if df.empty:
        return
    if 'date' not in df.columns:
        st.warning("campaign_performance must contain a 'date' column.")
        return
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    agg_level = st.selectbox("Aggregation level", options=['Daily', 'Weekly', 'Monthly'], index=2)
    channels = st.multiselect("Channels", options=df['channel'].dropna().unique().tolist(), default=df['channel'].dropna().unique().tolist())
    dff = df[(df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))]
    if channels:
        dff = dff[dff['channel'].isin(channels)]
    if agg_level == 'Daily':
        res = dff.groupby('date')['revenue'].sum().reset_index()
        fig = px.line(res, x='date', y='revenue', title='Daily Revenue Trend')
    elif agg_level == 'Weekly':
        res = dff.set_index('date').resample('W')['revenue'].sum().reset_index()
        fig = px.line(res, x='date', y='revenue', title='Weekly Revenue Trend')
    else:
        res = dff.set_index('date').resample('M')['revenue'].sum().reset_index()
        fig = px.line(res, x='date', y='revenue', title='Monthly Revenue Trend')
    st.plotly_chart(fig, use_container_width=True)

# Cumulative conversions area chart
def cumulative_conversions():
    df = safe_df('campaign')
    if df.empty:
        return
    region = st.selectbox("Region", options=['All'] + sorted(df['region'].dropna().unique().tolist()))
    dff = df.copy()
    if region != 'All':
        dff = dff[dff['region'] == region]
    dff = dff.groupby(['date', 'channel'])['conversions'].sum().reset_index().sort_values('date')
    dff['cum'] = dff.groupby('channel')['conversions'].cumsum()
    fig = px.area(dff, x='date', y='cum', color='channel', title='Cumulative Conversions by Channel')
    st.plotly_chart(fig, use_container_width=True)

# Histogram - Age distribution
def age_distribution():
    df = safe_df('customer')
    if df.empty:
        return
    bins = st.slider("Bins", min_value=5, max_value=100, value=20)
    segs = ['All'] + (df['segment'].dropna().unique().tolist() if 'segment' in df.columns else [])
    seg = st.selectbox("Segment", options=segs, index=0)
    dff = df.copy()
    if seg != 'All':
        dff = dff[dff['segment'] == seg]
    fig = px.histogram(dff, x='age', nbins=bins, title="Customer Age Distribution")
    st.plotly_chart(fig, use_container_width=True)

# Boxplot - LTV by segment
def ltv_by_segment():
    df = safe_df('customer')
    if df.empty:
        return
    show_points = st.checkbox("Show individual points", value=False)
    if 'segment' not in df.columns or 'ltv' not in df.columns:
        st.warning("Required columns 'segment' or 'ltv' missing in customer dataset.")
        return
    fig = px.box(df, x='segment', y='ltv', points='all' if show_points else 'outliers', title='LTV by Customer Segment')
    st.plotly_chart(fig, use_container_width=True)

# Violin - satisfaction by NPS
def satisfaction_violin():
    df = safe_df('customer')
    if df.empty:
        return
    split_col = 'acquisition_channel' if 'acquisition_channel' in df.columns else None
    if split_col:
        fig = px.violin(df, x='nps_category', y='satisfaction_score', color=split_col, box=True, points='outliers', title='Satisfaction by NPS and Channel')
    else:
        fig = px.violin(df, x='nps_category', y='satisfaction_score', box=True, points='outliers', title='Satisfaction by NPS')
    st.plotly_chart(fig, use_container_width=True)

# Scatter - Income vs LTV
def income_vs_ltv():
    df = safe_df('customer')
    if df.empty:
        return
    if 'income' not in df.columns or 'ltv' not in df.columns:
        st.warning("Columns 'income' or 'ltv' missing in customer dataset.")
        return
    show_trend = st.checkbox("Show trend line")
    fig = px.scatter(df, x='income', y='ltv', color='segment' if 'segment' in df.columns else None,
                     hover_data=['customer_id'] if 'customer_id' in df.columns else None,
                     title='Income vs LTV')
    if show_trend:
        sub = df.dropna(subset=['income', 'ltv'])
        if len(sub) > 1:
            model = LinearRegression()
            X = sub['income'].values.reshape(-1,1)
            y = sub['ltv'].values
            model.fit(X, y)
            xs = np.linspace(sub['income'].min(), sub['income'].max(), 100)
            ys = model.predict(xs.reshape(-1,1))
            fig.add_traces(go.Scatter(x=xs, y=ys, mode='lines', name='Trendline', line=dict(color='black')))
    st.plotly_chart(fig, use_container_width=True)

# Bubble chart - CTR vs Conversion Rate
def channel_bubble():
    df = safe_df('campaign')
    if df.empty:
        return
    agg = df.groupby('channel').agg({'ctr':'mean','conversion_rate':'mean','spend':'sum'}).reset_index().dropna()
    fig = px.scatter(agg, x='ctr', y='conversion_rate', size='spend', color='channel',
                     hover_data=['spend'], title='CTR vs Conversion Rate by Channel', size_max=60)
    st.plotly_chart(fig, use_container_width=True)

# Correlation heatmap
def correlation_heatmap():
    df = safe_df('corr')
    if df.empty:
        return
    try:
        fig = px.imshow(df.values, x=df.columns, y=df.index, color_continuous_scale='RdBu', zmin=-1, zmax=1, aspect='auto',
                        title='Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("Correlation matrix could not be rendered. Ensure it's square with row/column labels.")

# Calendar heatmap (GitHub-style)
def calendar_heatmap():
    df = safe_df('campaign')
    if df.empty:
        return
    metric = st.selectbox("Metric for calendar heatmap", options=['revenue', 'impressions'])
    if 'year' not in df.columns:
        st.warning("campaign_performance must include 'date' to compute year.")
        return
    years = sorted(df['year'].dropna().unique().tolist())
    if not years:
        st.warning("No year data available for calendar.")
        return
    year = st.selectbox("Year", options=years, index=len(years)-1)
    dff = df[df['year'] == year].groupby('date')[metric].sum().reset_index()
    if dff.empty:
        st.info("No daily data for selected year.")
        return
    dff['dow'] = dff['date'].dt.weekday
    dff['week'] = dff['date'].dt.isocalendar().week
    pivot = dff.pivot_table(index='dow', columns='week', values=metric, aggfunc='sum')
    fig = px.imshow(pivot, labels=dict(x='Week', y='Day of Week', color=metric), title=f'Calendar Heatmap ({year})')
    st.plotly_chart(fig, use_container_width=True)

# Donut chart - attribution
def donut_attribution():
    df = safe_df('attribution')
    if df.empty:
        return
    models = [c for c in df.columns if c != 'channel']
    if not models:
        st.warning("Attribution file must contain model columns (e.g., first_touch, last_touch).")
        return
    model = st.selectbox("Attribution Model", options=models)
    vals = df.set_index('channel')[model]
    fig = go.Figure(data=[go.Pie(labels=vals.index, values=vals.values, hole=.5)])
    fig.update_layout(title=f"Attribution: {model}", annotations=[dict(text=f"Total: {vals.sum():.0f}", showarrow=False)])
    st.plotly_chart(fig, use_container_width=True)

# Treemap - product hierarchy
def treemap_products():
    df = safe_df('product')
    if df.empty:
        return
    path = []
    for col in ['category','subcategory','product_name']:
        if col in df.columns:
            path.append(col)
    color = 'profit_margin' if 'profit_margin' in df.columns else None
    fig = px.treemap(df, path=path, values='sales', color=color, title='Product Sales Treemap')
    st.plotly_chart(fig, use_container_width=True)

# Sunburst - customer segmentation
def sunburst_segments():
    df = safe_df('customer')
    if df.empty:
        return
    path = []
    for col in ['region','city_tier','segment']:
        if col in df.columns:
            path.append(col)
    if not path:
        st.warning("Customer segmentation columns missing.")
        return
    fig = px.sunburst(df, path=path, title='Customer Segmentation Breakdown')
    st.plotly_chart(fig, use_container_width=True)

# Funnel chart
def funnel_chart():
    df = safe_df('funnel')
    if df.empty:
        return
    if 'stage' not in df.columns or 'visitors' not in df.columns:
        st.warning("Funnel data must contain 'stage' and 'visitors' columns.")
        return
    fig = px.funnel(df, x='visitors', y='stage', title='Conversion Funnel')
    st.plotly_chart(fig, use_container_width=True)

# Choropleth - state revenue
def choropleth_state():
    df = safe_df('geo')
    if df.empty:
        return
    candidates = [c for c in ['revenue','customers','market_penetration','yoy_growth'] if c in df.columns]
    if not candidates:
        st.warning("geographic_data must contain at least one of revenue/customers/market_penetration/yoy_growth")
        return
    metric = st.selectbox("Metric to show on map", options=candidates)
    try:
        loc_col = 'state_code' if 'state_code' in df.columns else 'state'
        fig = px.choropleth(df, locations=loc_col, color=metric,
                            locationmode='ISO-3' if loc_col == 'state_code' else None,
                            title=f"State-wise {metric}")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        if 'lat' in df.columns and 'lon' in df.columns:
            fig = px.scatter_geo(df, lat='lat', lon='lon', size='customers' if 'customers' in df.columns else None,
                                 color=metric, hover_name='state', title=f"State-wise {metric} (points)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Geographic visualization needs state codes or lat/lon in geographic_data.csv")

# Bubble map - store performance
def bubble_map_stores():
    df = safe_df('geo')
    if df.empty:
        return
    if 'lat' in df.columns and 'lon' in df.columns:
        size = 'store_count' if 'store_count' in df.columns else None
        color = 'satisfaction' if 'satisfaction' in df.columns else None
        fig = px.scatter_geo(df, lat='lat', lon='lon', size=size, color=color, hover_name='state', title='Store Performance')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("geographic_data.csv needs 'lat' and 'lon' to render bubble map.")

# Confusion matrix
def confusion_matrix_evaluation():
    df = safe_df('lead')
    if df.empty:
        return
    if 'actual_converted' not in df.columns or 'predicted_probability' not in df.columns:
        st.warning("Lead file must contain 'actual_converted' and 'predicted_probability'")
        return
    thresh = st.slider("Probability threshold", 0.0, 1.0, 0.5)
    preds = (df['predicted_probability'] >= thresh).astype(int)
    ct = pd.crosstab(df['actual_converted'], preds, rownames=['Actual'], colnames=['Predicted'])
    fig = px.imshow(ct.values, x=ct.columns.astype(str), y=ct.index.astype(str), text_auto=True, labels=dict(x='Predicted', y='Actual'),
                    title=f"Confusion Matrix (threshold={thresh:.2f})")
    st.plotly_chart(fig, use_container_width=True)

# ROC curve
def roc_evaluation():
    df = safe_df('lead')
    if df.empty:
        return
    if 'actual_converted' not in df.columns or 'predicted_probability' not in df.columns:
        st.warning("Lead file must contain 'actual_converted' and 'predicted_probability'")
        return
    fpr, tpr, thr = roc_curve(df['actual_converted'], df['predicted_probability'])
    roc_auc = auc(fpr, tpr)
    opt_idx = np.argmax(tpr - fpr)
    opt_thr = thr[opt_idx] if len(thr) > 0 else 0.5
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
    fig.add_trace(go.Scatter(x=[fpr[opt_idx]], y=[tpr[opt_idx]], mode='markers', name=f'Optimal thr={opt_thr:.2f}', marker=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"AUC = {roc_auc:.3f}; Suggested threshold ≈ {opt_thr:.2f}")

# Learning curve
def learning_curve_plot():
    df = safe_df('learning_curve')
    if df.empty:
        return
    show_conf = st.checkbox("Show confidence bands", value=True)
    fig = go.Figure()
    if 'train_score' in df.columns:
        fig.add_trace(go.Scatter(x=df['train_size'], y=df['train_score'], mode='lines+markers', name='Train'))
    if 'val_score' in df.columns:
        fig.add_trace(go.Scatter(x=df['train_size'], y=df['val_score'], mode='lines+markers', name='Validation'))
    if show_conf and 'train_std' in df.columns:
        fig.add_trace(go.Scatter(x=list(df['train_size']) + list(df['train_size'][::-1]),
                                 y=list(df['train_score'] + df['train_std']) + list((df['train_score'] - df['train_std'])[::-1]),
                                 fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
    st.plotly_chart(fig, use_container_width=True)

# Feature importance
def feature_importance():
    df = safe_df('feature_importance')
    if df.empty:
        return
    sort_asc = st.checkbox("Sort ascending", value=False)
    show_err = st.checkbox("Show error bars", value=True)
    plot_df = df.copy()
    plot_df = plot_df.sort_values('importance', ascending=sort_asc)
    fig = px.bar(plot_df, x='importance', y='feature', orientation='h', error_x='std' if show_err and 'std' in plot_df.columns else None,
                 title='Feature Importance')
    st.plotly_chart(fig, use_container_width=True)

# Sankey (Bonus)
def sankey_journey():
    df = safe_df('journey')
    if df.empty:
        return
    if not all(c in df.columns for c in ['source','target','count']):
        st.info("Sankey bonus requires columns: source, target, count in customer_journey.csv")
        return
    labels = list(pd.unique(df[['source','target']].values.ravel()))
    label_to_idx = {l:i for i,l in enumerate(labels)}
    sources = df['source'].map(label_to_idx)
    targets = df['target'].map(label_to_idx)
    values = df['count']
    link = dict(source=sources, target=targets, value=values)
    node = dict(label=labels, pad=20, thickness=20)
    fig = go.Figure(data=[go.Sankey(node=node, link=link)])
    fig.update_layout(title_text="Customer Journey Sankey", font_size=10)
    st.plotly_chart(fig, use_container_width=True)

# Precision-Recall curve (Bonus)
def pr_curve():
    df = safe_df('lead')
    if df.empty:
        return
    if 'actual_converted' not in df.columns or 'predicted_probability' not in df.columns:
        return
    precision, recall, _ = precision_recall_curve(df['actual_converted'], df['predicted_probability'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR Curve'))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Page layout and router
# ---------------------------
st.sidebar.title("NovaMart Dashboard")
page = st.sidebar.radio("Navigate", (
    "Executive Overview",
    "Campaign Analytics",
    "Customer Insights",
    "Product Performance",
    "Geographic Analysis",
    "Attribution & Funnel",
    "ML Model Evaluation"
))

if page == "Executive Overview":
    st.markdown('<div class="big-title">📈 Executive Overview</div>', unsafe_allow_html=True)
    st.markdown("High level KPIs and trendline.")
    kpi_cards()
    st.markdown("---")
    st.subheader("Revenue Trend")
    revenue_trend()
    st.markdown("---")
    st.subheader("Channel Performance")
    channel_performance()

elif page == "Campaign Analytics":
    st.markdown('<div class="big-title">📢 Campaign Analytics</div>', unsafe_allow_html=True)
    st.markdown("Time-series and comparison charts.")
    left, right = st.columns([2,1])
    with left:
        st.subheader("Revenue Trend")
        revenue_trend()
        st.subheader("Cumulative Conversions")
        cumulative_conversions()
        st.subheader("Calendar Heatmap")
        calendar_heatmap()
    with right:
        st.subheader("Channel Performance")
        channel_performance()
        st.subheader("Regional Performance by Quarter")
        regional_by_quarter()
        st.subheader("Campaign Spend Composition")
        stacked_campaign_spend()

elif page == "Customer Insights":
    st.markdown('<div class="big-title">🧑‍🤝‍🧑 Customer Insights</div>', unsafe_allow_html=True)
    st.subheader("Distribution Charts")
    c1, c2 = st.columns(2)
    with c1:
        age_distribution()
        ltv_by_segment()
    with c2:
        satisfaction_violin()
        income_vs_ltv()
    st.markdown("---")
    st.subheader("Relationship Charts")
    channel_bubble()

elif page == "Product Performance":
    st.markdown('<div class="big-title">🛒 Product Performance</div>', unsafe_allow_html=True)
    treemap_products()
    st.markdown("---")
    st.info("Additional category/regional views can be added here.")

elif page == "Geographic Analysis":
    st.markdown('<div class="big-title">🌍 Geographic Analysis</div>', unsafe_allow_html=True)
    choropleth_state()
    st.markdown("---")
    bubble_map_stores()

elif page == "Attribution & Funnel":
    st.markdown('<div class="big-title">🔄 Attribution & Funnel</div>', unsafe_allow_html=True)
    donut_attribution()
    st.markdown("---")
    funnel_chart()
    st.markdown("---")
    correlation_heatmap()

elif page == "ML Model Evaluation":
    st.markdown('<div class="big-title">🤖 ML Model Evaluation</div>', unsafe_allow_html=True)
    st.subheader("Confusion Matrix")
    confusion_matrix_evaluation()
    st.subheader("ROC Curve")
    roc_evaluation()
    st.markdown("---")
    st.subheader("Learning Curve")
    learning_curve_plot()
    st.markdown("---")
    st.subheader("Feature Importance")
    feature_importance()
    st.markdown("---")
    st.subheader("Bonus: Sankey (Customer Journey)")
    sankey_journey()
    st.subheader("Bonus: Precision-Recall Curve")
    pr_curve()

# Footer / help
st.sidebar.markdown("---")
st.sidebar.markdown("Built for: Masters of AI in Business — NovaMart")
st.sidebar.markdown("Author: Data Analyst")
st.sidebar.markdown("Tip: Place CSVs at same level as app.py (root) and re-run if charts are blank.")
