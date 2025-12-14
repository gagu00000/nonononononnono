import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(layout="wide")
st.title("🗺 PyDeck Map Test – Using My CSV")

# -----------------------------
# LOAD CSV
# -----------------------------
df = pd.read_csv("geographic_data.csv")
df.columns = df.columns.str.lower()

st.subheader("Detected columns in geographic_data.csv")
st.write(list(df.columns))
st.write(df.head())

# -----------------------------
# COLUMN MAPPING (SAFE)
# -----------------------------
lat_col = next((c for c in df.columns if c in ["lat", "latitude"]), None)
lon_col = next((c for c in df.columns if c in ["lon", "lng", "longitude"]), None)
store_col = next((c for c in df.columns if "store" in c), None)
sat_col = next((c for c in df.columns if "satisfaction" in c), None)

if not all([lat_col, lon_col, store_col, sat_col]):
    st.error(
        "CSV must contain latitude, longitude, store count, and satisfaction columns.\n\n"
        f"Detected → lat: {lat_col}, lon: {lon_col}, store: {store_col}, satisfaction: {sat_col}"
    )
    st.stop()

# -----------------------------
# PYDECK LAYER
# -----------------------------
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position=f'[{lon_col}, {lat_col}]',
    get_radius=f'{store_col} * 1200',
    get_fill_color=f'[255, (1 - {sat_col}/5)*255, 80, 160]',
    pickable=True,
)

# -----------------------------
# MAP VIEW (INDIA)
# -----------------------------
view_state = pdk.ViewState(
    latitude=df[lat_col].mean(),
    longitude=df[lon_col].mean(),
    zoom=4.2,
    pitch=45,
)

# -----------------------------
# DECK
# -----------------------------
deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/dark-v10",
    initial_view_state=view_state,
    layers=[layer],
    tooltip={
        "html": f"""
        <b>Stores:</b> {{{store_col}}}<br/>
        <b>Satisfaction:</b> {{{sat_col}}}
        """,
        "style": {"color": "white"}
    }
)

st.pydeck_chart(deck, use_container_width=True)

