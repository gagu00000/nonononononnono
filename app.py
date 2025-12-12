# Enhanced data loading and helper functions for NovaMart Dashboard

import streamlit as st
import pandas as pd
import numpy as np

# Constants
CSV_FILES = {
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

@st.cache_data
def safe_read_csv(path: str, parse_dates=None) -> pd.DataFrame:
    """Safely read CSV with error handling"""
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading {path}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_all() -> dict:
    """Load all CSV files with normalization"""
    data = {}
    
    # Load campaign data with date parsing
    data['campaign'] = safe_read_csv(CSV_FILES['campaign'], parse_dates=['date'])
    
    # Load other datasets
    for key in ['customer', 'product', 'lead', 'feature_importance', 
                'learning_curve', 'geo', 'attribution', 'funnel', 'journey', 'corr']:
        data[key] = safe_read_csv(CSV_FILES[key])
    
    # Enrich campaign dataframe
    if not data['campaign'].empty and 'date' in data['campaign'].columns:
        try:
            df = data['campaign']
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.strftime('%B')
            df['quarter'] = df['date'].dt.to_period('Q').astype(str)
            data['campaign'] = df
        except Exception as e:
            st.warning(f"Error enriching campaign data: {e}")
    
    # Normalize learning curve columns
    if not data['learning_curve'].empty:
        lc = data['learning_curve']
        rename_map = {}
        
        if 'training_size' in lc.columns and 'train_size' not in lc.columns:
            rename_map['training_size'] = 'train_size'
        if 'validation_score' in lc.columns and 'val_score' not in lc.columns:
            rename_map['validation_score'] = 'val_score'
        
        if rename_map:
            data['learning_curve'] = lc.rename(columns=rename_map)
    
    # Normalize geographic columns
    if not data['geo'].empty:
        geo = data['geo']
        
        if 'lat' in geo.columns and 'latitude' not in geo.columns:
            geo = geo.rename(columns={'lat': 'latitude'})
        if 'lon' in geo.columns and 'longitude' not in geo.columns:
            geo = geo.rename(columns={'lon': 'longitude'})
        
        data['geo'] = geo
    
    return data

def df_or_warn(key: str, data_dict: dict) -> pd.DataFrame:
    """Get dataframe or show warning if missing"""
    df = data_dict.get(key)
    if df is None or df.empty:
        st.warning(f"📊 Dataset `{key}` missing or empty. Upload `{CSV_FILES[key]}` to enable related charts.")
        return pd.DataFrame()
    return df.copy()

def validate_columns(df: pd.DataFrame, required_cols: list, dataset_name: str) -> bool:
    """Validate that required columns exist in dataframe"""
    if df.empty:
        return False
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.warning(f"⚠️ {dataset_name} missing required columns: {', '.join(missing)}")
        return False
    return True

def money(x) -> str:
    """Format number as Indian Rupees"""
    if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)):
        return f"₹{x:,.0f}"
    return "N/A"

def safe_date_range(date_range: tuple) -> tuple:
    """Ensure dates are in correct order"""
    dates = [pd.to_datetime(d) for d in date_range]
    return tuple(sorted(dates))

# Example enhanced function with all improvements
def revenue_trend_enhanced(data_dict: dict):
    """Enhanced revenue trend with better error handling"""
    st.subheader("Revenue Trend Over Time")
    
    df = df_or_warn('campaign', data_dict)
    if not validate_columns(df, ['date', 'revenue'], 'campaign_performance.csv'):
        return
    
    # Date range selector with safe handling
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.date_input(
        "Date range", 
        value=(min_date, max_date), 
        min_value=min_date, 
        max_value=max_date, 
        key="rt_dates"
    )
    
    # Ensure proper date order
    start_date, end_date = safe_date_range(date_range)
    
    # Aggregation level
    agg_level = st.selectbox("Aggregation level", ['Daily', 'Weekly', 'Monthly'], index=2)
    
    # Channel filter
    channels_available = df['channel'].dropna().unique().tolist() if 'channel' in df.columns else []
    channels = st.multiselect("Channels", options=channels_available, default=channels_available)
    
    # Filter data
    dff = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    if channels:
        dff = dff[dff['channel'].isin(channels)]
    
    if dff.empty:
        st.info("No data available for selected filters")
        return
    
    # Aggregate by level
    if agg_level == 'Daily':
        res = dff.groupby('date')['revenue'].sum().reset_index()
    elif agg_level == 'Weekly':
        res = dff.set_index('date').resample('W')['revenue'].sum().reset_index()
    else:
        res = dff.set_index('date').resample('M')['revenue'].sum().reset_index()
    
    # Create chart
    import plotly.express as px
    fig = px.line(res, x='date', y='revenue', title=f"{agg_level} Revenue Trend")
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Add download button
    st.download_button(
        label="📥 Download Data",
        data=res.to_csv(index=False),
        file_name=f"revenue_trend_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
