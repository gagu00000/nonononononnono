import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import json

# -----------------------------------------------------------
# GLOBAL DARK THEME FOR PLOTLY (fixes invisible charts issue)
# -----------------------------------------------------------
pio.templates["streamlit_dark"] = pio.templates["plotly_dark"]
pio.templates["streamlit_dark"].layout.update(
    {
        "paper_bgcolor": "#111111",
        "plot_bgcolor": "#111111",
        "font": {"color": "#ffffff"},
        "geo": {
            "bgcolor": "#111111",
            "lakecolor": "#111111",
            "landcolor": "#222222",
            "subunitcolor": "#444444",
        },
        "coloraxis": {"colorbar": {"tickcolor": "#ffffff"}},
    }
)
pio.templates.default = "streamlit_dark"

# -----------------------------------------------------------
# STREAMLIT PAGE CONFIG + DARK MODE CSS 
# -----------------------------------------------------------
st.set_page_config(layout="wide", page_title="NOVA Marketing Dashboard")

dark_css = """
<style>
body, .stApp {
    background-color: #0E1117 !important;
    color: white !important;
}
section.main > div {
    background-color: #0E1117 !important;
}
[data-testid="stSidebar"] {
    background-color: #161A22 !important;
}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# -----------------------------------------------------------
# EMBEDDED INDIA GEOJSON (ready to use, no download needed)
# -----------------------------------------------------------
india_geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "id": "IN-MH",
            "properties": {"state": "Maharashtra"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[72.6, 18.9], [73.8, 19.0], [75.3, 17.9], [73.1, 16.1], [72.6, 18.9]]]
            },
        },
        {
            "type": "Feature",
            "id": "IN-GJ",
            "properties": {"state": "Gujarat"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[68.1, 22.8], [71.5, 23.5], [72.5, 21.2], [69.5, 20.7], [68.1, 22.8]]]
            },
        },
        {
            "type": "Feature",
            "id": "IN-KA",
            "properties": {"state": "Karnataka"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[74.0, 15.6], [76.0, 16.5], [77.0, 14.0], [75.0, 13.0], [74.0, 15.6]]]
            },
        },
        {
            "type": "Feature",
            "id": "IN-TN",
            "properties": {"state": "Tamil Nadu"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[78.0, 12.8], [80.2, 13.5], [80.3, 10.5], [77.5, 10.0], [78.0, 12.8]]]
            },
        },
        {
            "type": "Feature",
            "id": "IN-DL",
            "properties": {"state": "Delhi"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[77.0, 28.5], [77.4, 28.6], [77.5, 28.3], [77.1, 28.2], [77.0, 28.5]]]
            },
        }
    ],
}

# -----------------------------------------------------------
# LOAD ALL DATASETS (from root folder, same as app.py)
# -----------------------------------------------------------
@st.cache_data
def load_data():
    dfs = {}
    files = [
        "campaign_performance.csv",
        "channel_attribution.csv",
        "correlation_matrix.csv",
        "customer_data.csv",
        "customer_journey.csv",
        "feature_importance.csv",
        "funnel_data.csv",
        "geographic_data.csv",
        "lead_scoring_results.csv",
        "learning_curve.csv",
        "product_sales.csv"
    ]
    for f in files:
        try:
            dfs[f.replace(".csv", "")] = pd.read_csv(f)
        except:
            st.error(f"❌ Missing file: {f}")

    return dfs


dfs = load_data()

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to", 
    ["Overview", "Campaign Performance", "Channel Attribution",
     "Funnel Analysis", "Customer Insights",
     "Geographic Analysis", "Model Insights"]
)

# -----------------------------------------------------------
# PAGE: OVERVIEW
# -----------------------------------------------------------
if page == "Overview":
    st.title("📈 NOVA Marketing Intelligence Dashboard")

    col1, col2, col3 = st.columns(3)
    try:
        sales = dfs["product_sales"]["sales"].sum()
        leads = dfs["lead_scoring_results"]["lead_score"].count()
        customers = dfs["customer_data"]["customer_id"].nunique()

        col1.metric("Total Sales", f"₹{sales:,.0f}")
        col2.metric("Total Leads", f"{leads:,}")
        col3.metric("Customers", f"{customers:,}")

    except:
        st.warning("Some datasets not loaded fully.")

# -----------------------------------------------------------
# PAGE: GEOGRAPHIC ANALYSIS (fixed version)
# -----------------------------------------------------------
elif page == "Geographic Analysis":
    st.title("🗺 Geographic Performance")

    try:
        geo_df = dfs["geographic_data"]

        fig = px.choropleth(
            geo_df,
            geojson=india_geojson,
            featureidkey="properties.state",
            locations="state",
            color="sales",
            color_continuous_scale="Viridis",
            title="Sales by State",
        )
        fig.update_geos(fitbounds="locations", visible=False)

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------------------------------------
# PAGE: MODEL INSIGHTS
# -----------------------------------------------------------
elif page == "Model Insights":
    st.title("🤖 Model Insights")

    try:
        fi = dfs["feature_importance"]
        fig = px.bar(fi, x="feature", y="importance", title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Feature importance data missing.")

