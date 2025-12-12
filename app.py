"""
app.py - NovaMart Marketing Analytics Dashboard (final)
- Place all 11 CSVs in the same folder as this file (root).
- Default choropleth metric: total_revenue (you selected option 1).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LinearRegression
import urllib.request
import json
from io import BytesIO
# =========================================
# 🔥 GLOBAL DARK THEME FIX (Force Dark Mode)
# =========================================
import streamlit as st

st.markdown("""
<style>

/* Global App Background */
.stApp {
    background-color: #111 !important;
    color: #f0f0f0 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1a1a1a !important;
    color: #f0f0f0 !important;
}

/* Metric Cards */
div[data-testid="metric-container"] {
    background-color: #222 !important;
    padding: 15px !important;
    border-radius: 10px !important;
    color: #f0f0f0 !important;
}

/* All Cards / Containers */
div.css-1r6slb0, div.css-12w0qpk, .stMarkdown {
    background-color: #111 !important;
    color: #ffffff !important;
}

/* Inputs */
.stSelectbox, .stTextInput, .stNumberInput, .stDateInput, .stMultiSelect {
    background-color: #222 !important;
    color: #fff !important;
}

.stSelectbox > div > div {
    background-color: #222 !important;
    color: #fff !important;
}

/* Plotly chart background */
.js-plotly-plot .plotly, .plot-container {
    background-color: #111 !important;
    color: #fff !important;
}

/* Remove white blocks behind charts */
svg.main-svg {
    background-color: #111 !important;
}

/* Fix tables */
.dataframe {
    background-color: #111 !important;
    color: #fff !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Theme & styling
# ---------------------------
PRIMARY = "#0B3D91"   # deep blue
ACCENT = "#F4B400"    # gold
BG = "#F5F7FA"

st.set_page_config(page_title="NovaMart Marketing Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG}; }}
    .block-container, .main, .reportview-container {{ color: #0A0A0A !important; }}
    .big-title {{ font-size:34px; font-weight:700; color:{PRIMARY} !important; background: transparent !important; padding: 4px 6px; border-radius: 6px; }}
    /* Make Streamlit metrics readable */
    div[data-testid="metric-container"] .stMetricLabel, div[data-testid="metric-container"] .stMetricValue,
    div[data-testid="metric-container"] .css-1v0mbdj, div[data-testid="metric-container"] .css-1v0mbdj * {{
        color: #0A0A0A !important;
        background: transparent !important;
    }}
    /* Plotly charts background */
    .stPlotlyChart > div {{ background: white !important; border-radius: 6px; padding: 6px; }}
    /* Form controls */
    .stSelectbox, .stMultiSelect, .stSlider, .stDateInput, .stRadio, .stTextInput {{
        color: #0A0A0A !important;
    }}
    /* Info/warning text */
    .stInfo, .stWarning, .stAlert {{ color: #0A0A0A !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Data loading (root)
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

data = load_all()

# ---------------------------
# Helpers
# ---------------------------
def warn_missing(name):
    st.warning(f"Dataset `{name}` not found or empty. Upload `{name}.csv` beside app.py to enable related charts.")

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
# Visualizations
# ---------------------------

def kpi_cards():
    df = safe_df('campaign')
    cust = safe_df('customer')
    c1, c2, c3, c4 = st.columns(4)
    if df.empty:
        c1.metric("Total Revenue", "N/A")
        c2.metric("Total Conversions", "N/A")
        c3.metric("Total Spend", "N/A")
        c4.metric("ROAS", "N/A")
    else:
        total_rev = df['revenue'].sum()
        total_conv = df['conversions'].sum()
        total_spend = df['spend'].sum() if 'spend' in df.columns else 0
        roas = total_rev / total_spend if total_spend else np.nan
        c1.metric("Total Revenue", currency_fmt(total_rev))
        c2.metric("Total Conversions", f"{int(total_conv):,}")
        c3.metric("Total Spend", currency_fmt(total_spend))
        c4.metric("ROAS", f"{roas:.2f}" if not np.isnan(roas) else "N/A")
    c4.metric("Customer Count", f"{cust.shape[0]:,}" if not cust.empty else "N/A")

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

def revenue_trend():
    df = safe_df('campaign')
    if df.empty:
        return
    if 'date' not in df.columns:
        st.warning("campaign_performance must contain 'date' column.")
        return
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    agg_level = st.selectbox("Aggregation level", ['Daily', 'Weekly', 'Monthly'], index=2)
    channels = st.multiselect("Channels", options=df['channel'].dropna().unique().tolist(), default=df['channel'].dropna().unique().tolist())
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

def age_distribution():
    df = safe_df('customer')
    if df.empty:
        return
    bins = st.slider("Bins", 5, 100, 20)
    segs = ['All'] + (df['segment'].dropna().unique().tolist() if 'segment' in df.columns else [])
    seg = st.selectbox("Segment", options=segs, index=0)
    dff = df.copy()
    if seg != 'All':
        dff = dff[dff['segment'] == seg]
    fig = px.histogram(dff, x='age', nbins=bins, title="Customer Age Distribution")
    st.plotly_chart(fig, use_container_width=True)

def ltv_by_segment():
    df = safe_df('customer')
    if df.empty:
        return
    if 'segment' not in df.columns or 'ltv' not in df.columns:
        st.warning("Customer data missing 'segment' or 'ltv' columns.")
        return
    show_points = st.checkbox("Show individual points", value=False)
    fig = px.box(df, x='segment', y='ltv', points='all' if show_points else 'outliers', title='LTV by Segment')
    st.plotly_chart(fig, use_container_width=True)

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

def income_vs_ltv():
    df = safe_df('customer')
    if df.empty:
        return
    if 'income' not in df.columns or 'ltv' not in df.columns:
        st.warning("Customer data must include 'income' and 'ltv' columns.")
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

def channel_bubble():
    df = safe_df('campaign')
    if df.empty:
        return
    agg = df.groupby('channel').agg({'ctr':'mean','conversion_rate':'mean','spend':'sum'}).reset_index().dropna()
    fig = px.scatter(agg, x='ctr', y='conversion_rate', size='spend', color='channel',
                     hover_data=['spend'], title='CTR vs Conversion Rate by Channel', size_max=60)
    st.plotly_chart(fig, use_container_width=True)

def correlation_heatmap():
    df = safe_df('corr')
    if df.empty:
        return
    try:
        fig = px.imshow(df.values, x=df.columns, y=df.index, color_continuous_scale='RdBu', zmin=-1, zmax=1, aspect='auto',
                        title='Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("Correlation matrix not renderable. Ensure it's a square matrix with row/column labels.")

def calendar_heatmap():
    df = safe_df('campaign')
    if df.empty:
        return
    if 'year' not in df.columns:
        st.warning("campaign_performance must include 'date' to compute yearly calendar heatmap.")
        return
    metric = st.selectbox("Metric for calendar heatmap", options=['revenue', 'impressions'])
    years = sorted(df['year'].dropna().unique().tolist())
    year = st.selectbox("Year", options=years, index=len(years)-1 if years else 0)
    dff = df[df['year'] == year].groupby('date')[metric].sum().reset_index()
    if dff.empty:
        st.info("No daily data for selected year.")
        return
    dff['dow'] = dff['date'].dt.weekday
    dff['week'] = dff['date'].dt.isocalendar().week
    pivot = dff.pivot_table(index='dow', columns='week', values=metric, aggfunc='sum')
    fig = px.imshow(pivot, labels=dict(x='Week', y='Day of Week', color=metric), title=f'Calendar Heatmap ({year})')
    st.plotly_chart(fig, use_container_width=True)

def donut_attribution():
    df = safe_df('attribution')
    if df.empty:
        return
    models = [c for c in df.columns if c != 'channel']
    if not models:
        st.warning("Attribution CSV must have 'channel' plus model columns.")
        return
    model = st.selectbox("Attribution Model", options=models)
    vals = df.set_index('channel')[model]
    fig = go.Figure(data=[go.Pie(labels=vals.index, values=vals.values, hole=.5)])
    fig.update_layout(title=f"Attribution: {model}", annotations=[dict(text=f"Total: {vals.sum():.0f}", showarrow=False)])
    st.plotly_chart(fig, use_container_width=True)

def treemap_products():
    df = safe_df('product')
    if df.empty:
        return
    path = [c for c in ['category', 'subcategory', 'product_name'] if c in df.columns]
    if not path:
        st.warning("product_sales needs category/subcategory/product_name columns.")
        return
    fig = px.treemap(df, path=path, values='sales', color='profit_margin' if 'profit_margin' in df.columns else None, title='Product Sales Treemap')
    st.plotly_chart(fig, use_container_width=True)

def sunburst_segments():
    df = safe_df('customer')
    if df.empty:
        return
    path = [c for c in ['region','city_tier','segment'] if c in df.columns]
    if not path:
        st.warning("customer_data must have segmentation fields (region, city_tier, segment).")
        return
    fig = px.sunburst(df, path=path, title='Customer Segmentation Breakdown')
    st.plotly_chart(fig, use_container_width=True)

def funnel_chart():
    df = safe_df('funnel')
    if df.empty:
        return
    if 'stage' not in df.columns or 'visitors' not in df.columns:
        st.warning("funnel_data must have 'stage' and 'visitors' columns.")
        return
    fig = px.funnel(df, x='visitors', y='stage', title='Conversion Funnel')
    st.plotly_chart(fig, use_container_width=True)

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

def confusion_matrix_evaluation():
    df = safe_df('lead')
    if df.empty:
        return
    if 'actual_converted' not in df.columns or 'predicted_probability' not in df.columns:
        st.warning("lead_scoring_results must contain 'actual_converted' and 'predicted_probability'")
        return
    thresh = st.slider("Probability threshold", 0.0, 1.0, 0.5)
    preds = (df['predicted_probability'] >= thresh).astype(int)
    ct = pd.crosstab(df['actual_converted'], preds, rownames=['Actual'], colnames=['Predicted'])
    fig = px.imshow(ct.values, x=ct.columns.astype(str), y=ct.index.astype(str), text_auto=True, labels=dict(x='Predicted', y='Actual'),
                    title=f"Confusion Matrix (threshold={thresh:.2f})")
    st.plotly_chart(fig, use_container_width=True)

def roc_evaluation():
    df = safe_df('lead')
    if df.empty:
        return
    if 'actual_converted' not in df.columns or 'predicted_probability' not in df.columns:
        st.warning("lead_scoring_results must contain 'actual_converted' and 'predicted_probability'")
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

def sankey_journey():
    df = safe_df('journey')
    if df.empty:
        return
    if not all(c in df.columns for c in ['source','target','count']):
        st.info("customer_journey must include source/target/count for Sankey.")
        return
    labels = list(pd.unique(df[['source','target']].values.ravel()))
    mapping = {l:i for i,l in enumerate(labels)}
    fig = go.Figure(data=[go.Sankey(node=dict(label=labels), link=dict(
        source=df['source'].map(mapping), target=df['target'].map(mapping), value=df['count']))])
    fig.update_layout(title_text="Customer Journey Sankey", font_size=10)
    st.plotly_chart(fig, use_container_width=True)

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
# Geo helpers & choropleth (default metric: total_revenue)
# ---------------------------
def _load_local_geojson(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def _download_geojson(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            if response.status == 200:
                data = response.read()
                return json.loads(data.decode('utf-8'))
    except Exception:
        return None

def choropleth_state():
    df = safe_df('geo')
    if df.empty:
        return

    # default preference: total_revenue (user selected option 1), but allow other metrics if present
    candidates = [c for c in ['total_revenue','total_customers','market_penetration','yoy_growth'] if c in df.columns]
    if not candidates:
        st.warning("geographic_data.csv must have at least one metric: total_revenue/total_customers/market_penetration/yoy_growth")
        return
    # default to total_revenue if present
    default_metric = 'total_revenue' if 'total_revenue' in candidates else candidates[0]
    metric = st.selectbox("Metric to show on map", options=candidates, index=candidates.index(default_metric))

    # 1) If lat/lon present -> scatter_geo
    if 'latitude' in df.columns and 'longitude' in df.columns:
        st.info("Rendering bubble map from latitude/longitude.")
        size_col = 'store_count' if 'store_count' in df.columns else None
        color_col = 'customer_satisfaction' if 'customer_satisfaction' in df.columns else metric
        fig = px.scatter_geo(
            df,
            lat='latitude',
            lon='longitude',
            size=size_col,
            color=color_col,
            hover_name='state' if 'state' in df.columns else None,
            title=f"Store Performance: {metric}",
            projection="natural earth"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        return

    # 2) Try local geojson file (repo root: india_states_geojson.json)
    local_geo = _load_local_geojson("india_states_geojson.json")
    geojson = None
    geojson_source = None
    if local_geo:
        geojson = local_geo
        geojson_source = "local file (india_states_geojson.json)"
    else:
        # 3) Attempt to download common India geojsons
        st.info("Attempting to download India GeoJSON (best-effort).")
        urls = [
            "https://raw.githubusercontent.com/datameet/maps/master/geojson/state_india.geojson",
            "https://raw.githubusercontent.com/plotly/datasets/master/india_states.geojson",
            "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
        ]
        for u in urls:
            geo = _download_geojson(u)
            if geo:
                geojson = geo
                geojson_source = u
                break

    if geojson:
        try:
            st.success(f"GeoJSON loaded from: {geojson_source}")
            # detect property key that holds state name
            sample = geojson.get('features', [None])[0]
            prop_keys = []
            if sample and 'properties' in sample:
                prop_keys = list(sample['properties'].keys())
            candidates_keys = [k for k in ['st_nm', 'NAME', 'name', 'STATE', 'state'] if k in prop_keys]
            prop_key = candidates_keys[0] if candidates_keys else (prop_keys[0] if prop_keys else None)
            if prop_key is None:
                st.warning("GeoJSON loaded but no properties found; falling back to bar chart.")
                raise ValueError("No property keys in geojson features")

            # Normalize function for better matching
            def normalize(s):
                if pd.isna(s):
                    return ""
                return str(s).strip().lower().replace('.', '').replace('&', 'and')

            df_map = df.copy()
            df_map['state_norm'] = df_map['state'].astype(str).apply(normalize)

            # Build mapping: normalized geojson name -> feature id or name
            feature_map = {}
            for i, feat in enumerate(geojson.get('features', [])):
                name = feat.get('properties', {}).get(prop_key, '')
                feature_map[normalize(name)] = feat.get('id', name)  # prefer id, else name

            df_map['featureid'] = df_map['state_norm'].map(feature_map)

            missing = df_map['featureid'].isna().sum()
            if missing > 0:
                st.warning(f"{missing} states could not be auto-matched. Matches may still work if geojson properties align; otherwise add 'india_states_geojson.json' to repo matched to your state names.")

            # Use choropleth_mapbox with featureidkey on properties (fallback robust)
            center = {"lat": 22.5937, "lon": 78.9629}
            fig = px.choropleth_mapbox(df_map, geojson=geojson, locations='featureid', color=metric,
                                       featureidkey=f"properties.{prop_key}",
                                       hover_name='state',
                                       mapbox_style="carto-positron",
                                       center=center, zoom=3.6,
                                       title=f"India: {metric} by State")
            fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=650)
            st.plotly_chart(fig, use_container_width=True)
            return
        except Exception as e:
            st.warning("Choropleth attempt failed; showing fallback visualization.")
            st.exception(e)

    # Final fallback: bar chart by state
    st.info("Falling back to bar chart showing top states by metric.")
    if 'state' in df.columns:
        bar = df.groupby('state')[metric].sum().reset_index().sort_values(metric, ascending=False).head(20)
        fig = px.bar(bar, x=metric, y='state', orientation='h', title=f"Top States by {metric}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("geographic_data.csv missing both lat/lon and state columns. Add 'state' or lat/lon to enable maps.")

# ---------------------------
# Router / Layout
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
        if not safe_df('campaign').empty:
            years = sorted(safe_df('campaign')['year'].dropna().unique().tolist())
            if years:
                sel_year = st.selectbox("Select Year (for regional view)", options=years, index=len(years)-1)
                tmp = safe_df('campaign')[safe_df('campaign')['year']==sel_year].groupby(['region','quarter'])['revenue'].sum().reset_index()
                fig = px.bar(tmp, x='region', y='revenue', color='quarter', barmode='group', title=f"Revenue by Region - {sel_year}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No year data for regional performance.")
        else:
            st.info("Upload campaign_performance.csv to enable regional charts.")
        st.subheader("Campaign Spend Composition")
        df = safe_df('campaign')
        if not df.empty:
            mode = st.radio("Spend view", ['Absolute','100% (Normalized)'])
            tmp = df.groupby(['month','campaign_type'], sort=False)['spend'].sum().reset_index()
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

elif page == "Customer Insights":
    st.markdown('<div class="big-title">🧑‍🤝‍🧑 Customer Insights</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age_distribution()
        ltv_by_segment()
    with c2:
        satisfaction_violin()
        income_vs_ltv()
    st.markdown("---")
    st.subheader("Channel Performance Matrix")
    channel_bubble()

elif page == "Product Performance":
    st.markdown('<div class="big-title">🛒 Product Performance</div>', unsafe_allow_html=True)
    treemap_products()
    st.markdown("---")
    st.info("Add more product/regional breakdown charts as needed.")

elif page == "Geographic Analysis":
    st.markdown('<div class="big-title">🌍 Geographic Analysis</div>', unsafe_allow_html=True)
    choropleth_state()
    st.markdown("---")
    st.subheader("Store Performance (Bubble Map if latitude/longitude present)")
    if not safe_df('geo').empty:
        if 'latitude' in safe_df('geo').columns and 'longitude' in safe_df('geo').columns:
            fig = px.scatter_geo(safe_df('geo'), lat='latitude', lon='longitude', size='store_count' if 'store_count' in safe_df('geo').columns else None,
                                 color='customer_satisfaction' if 'customer_satisfaction' in safe_df('geo').columns else None, hover_name='state', projection="natural earth",
                                 title='Store Performance')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload latitude/longitude in geographic_data.csv to enable bubble map; otherwise use choropleth or bar fallback.")

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

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built for: Masters of AI in Business — NovaMart")
st.sidebar.markdown("Author: Data Analyst")
st.sidebar.markdown("Tip: Place CSVs at same level as app.py (root) and restart the app if charts are blank.")
