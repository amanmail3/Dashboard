# =========================================================
# AI REAL ESTATE COPILOT — FINAL (PRICE FIX + UNIT TOGGLE)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
from openai import OpenAI

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Real Estate Copilot",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# OPENAI CONFIG (USE STREAMLIT SECRETS / ENV VAR)
# ---------------------------------------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------------------------------------------------
# GLOBAL STYLES (GLASS + NEON)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    div[data-testid="metric-container"],
    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(14px);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 0 25px rgba(0,198,255,0.4);
        padding: 16px;
        margin-bottom: 16px;
        transition: all 0.4s ease;
    }
    div[data-testid="stPlotlyChart"]:hover {
        box-shadow: 0 0 40px rgba(0,198,255,0.8);
        transform: scale(1.01);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Real.Estate.Data.V21.csv")

df = load_data()

# ---------------------------------------------------------
# PRICE CLEANING (CRITICAL FIX)
# ---------------------------------------------------------
def parse_price(price):
    if pd.isna(price):
        return np.nan

    price = str(price).replace(",", "").replace("₹", "").lower().strip()

    try:
        if "cr" in price:
            return float(re.findall(r"\d+\.?\d*", price)[0]) * 1e7
        if "l" in price:
            return float(re.findall(r"\d+\.?\d*", price)[0]) * 1e5
        return float(re.findall(r"\d+\.?\d*", price)[0])
    except:
        return np.nan

df["Price_num"] = df["Price"].apply(parse_price)
df = df.dropna(subset=["Price_num"])

# ---------------------------------------------------------
# PRETTY PRICE FORMATTER
# ---------------------------------------------------------
def format_price(value, unit):
    if unit == "Cr":
        return f"₹{value/1e7:.2f} Cr"
    else:
        return f"₹{value/1e5:.1f} L"

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
def extract_bhk(text):
    match = re.search(r"(\d+)\s*BHK", str(text))
    return int(match.group(1)) if match else np.nan

df["BHK"] = df["Property Title"].apply(extract_bhk)

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.title("⚙️ Controls")

price_unit = st.sidebar.radio("💰 Price Unit", ["Cr", "L"], horizontal=True)

locations = st.sidebar.multiselect(
    "📍 Location",
    sorted(df["Location"].unique()),
    sorted(df["Location"].unique())
)

price_range = st.sidebar.slider(
    "Price Range",
    int(df["Price_num"].min()),
    int(df["Price_num"].max()),
    (
        int(df["Price_num"].min()),
        int(df["Price_num"].max())
    )
)

# ---------------------------------------------------------
# FILTER DATA
# ---------------------------------------------------------
filtered_df = df[
    (df["Location"].isin(locations)) &
    (df["Price_num"].between(price_range[0], price_range[1]))
]

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🏙️ AI Real Estate Copilot")

# ---------------------------------------------------------
# KPI METRICS
# ---------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("Listings", len(filtered_df))
c2.metric(
    "Avg Price",
    format_price(filtered_df["Price_num"].mean(), price_unit)
)
c3.metric(
    "Max Price",
    format_price(filtered_df["Price_num"].max(), price_unit)
)
c4.metric(
    "Avg ₹ / Sqft",
    f"₹{int(filtered_df['Price_per_SQFT'].mean()):,}"
)

st.divider()

# ---------------------------------------------------------
# PRICE vs AREA CHART
# ---------------------------------------------------------
st.subheader("📊 Price vs Area")

fig = px.scatter(
    filtered_df,
    x="Total_Area",
    y="Price_num",
    color="Location",
    size="Price_per_SQFT",
    hover_name="Property Title",
    labels={"Price_num": "Price (INR)"},
    template="plotly_dark"
)
fig.update_layout(transition_duration=500)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# AI CHART NARRATION
# ---------------------------------------------------------
@st.cache_data
def narrate_chart(df):
    summary = df[["Total_Area", "Price_num", "Price_per_SQFT"]].describe().to_string()
    prompt = f"Explain this real estate price vs area chart:\n{summary}"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120
    )
    return res.choices[0].message.content

with st.expander("🧠 Auto Chart Explanation"):
    st.markdown(narrate_chart(filtered_df))

# ---------------------------------------------------------
# DATA TABLE (WITH PRETTY PRICE)
# ---------------------------------------------------------
st.subheader("📋 Property Explorer")

table_df = filtered_df.copy()
table_df["Pretty Price"] = table_df["Price_num"].apply(
    lambda x: format_price(x, price_unit)
)

st.dataframe(
    table_df[[
        "Property Title",
        "Location",
        "Pretty Price",
        "Total_Area",
        "Price_per_SQFT",
        "Baths",
        "Balcony"
    ]],
    use_container_width=True
)

# ---------------------------------------------------------
# CHAT WITH YOUR DATA
# ---------------------------------------------------------
st.divider()
st.subheader("💬 Chat with your data")

question = st.text_input("Ask a question about the current selection")

if st.button("Ask AI"):
    summary = filtered_df.describe(include="all").to_string()
    prompt = f"Answer using this dataset summary:\n{summary}\nQuestion: {question}"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    st.success(res.choices[0].message.content)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.caption("AI Real Estate Copilot • Streamlit Cloud • Production Ready")
