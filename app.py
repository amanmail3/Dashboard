# =========================================================
# AI REAL ESTATE COPILOT — STREAMLIT APP (NEON + SPOTLIGHT)
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
# GLASSMORPHISM + NEON + ANIMATIONS (GLOBAL CSS)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }

    /* ---------- Glass cards ---------- */
    div[data-testid="metric-container"],
    div[data-testid="stDataFrame"],
    div[data-testid="stPlotlyChart"] {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(14px);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        padding: 16px;
        margin-bottom: 16px;
        animation: fadeIn 0.6s ease-in;
        transition: all 0.4s ease;
    }

    /* ---------- 🌈 Neon hover effect (charts) ---------- */
    div[data-testid="stPlotlyChart"]:hover {
        box-shadow:
            0 0 15px rgba(0,198,255,0.6),
            0 0 30px rgba(0,114,255,0.5),
            0 0 60px rgba(0,114,255,0.4);
        transform: scale(1.01);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: rgba(15,32,39,0.9);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* ---------- Buttons ---------- */
    button[kind="primary"] {
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    button[kind="primary"]:hover {
        transform: scale(1.07);
        box-shadow: 0 0 25px rgba(0,198,255,0.8);
    }

    /* ---------- Spotlight overlay ---------- */
    .spotlight {
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.65);
        z-index: 999;
        animation: fadeIn 0.4s ease-in;
    }

    .spotlight-box {
        position: relative;
        z-index: 1000;
        box-shadow:
            0 0 25px rgba(0,255,200,0.8),
            0 0 60px rgba(0,255,200,0.5);
        border-radius: 18px;
        padding: 12px;
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(14px);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# OPENAI CONFIG (PASTE KEY HERE)
# ---------------------------------------------------------
client = OpenAI(
    api_key="OPENAI_API_KEY"
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Real.Estate.Data.V21.csv")

df = load_data()

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
def extract_bhk(text):
    match = re.search(r"(\\d+)\\s*BHK", str(text))
    return int(match.group(1)) if match else np.nan

def infer_property_type(text):
    text = str(text).lower()
    return "House" if ("villa" in text or "house" in text) else "Flat"

df["BHK"] = df["Property Title"].apply(extract_bhk)
df["Property_Type"] = df["Property Title"].apply(infer_property_type)

# ---------------------------------------------------------
# SIDEBAR — DEMO MODE + FILTERS
# ---------------------------------------------------------
st.sidebar.title("🚀 Demo Controls")
demo_mode = st.sidebar.toggle("One-Click Demo Mode")

st.sidebar.markdown("---")
st.sidebar.title("🔍 Filters")

if demo_mode:
    locations = df["Location"].value_counts().head(2).index.tolist()
    price_range = (
        int(df["Price"].quantile(0.25)),
        int(df["Price"].quantile(0.75))
    )
else:
    locations = st.sidebar.multiselect(
        "Location",
        df["Location"].unique(),
        df["Location"].unique()
    )

    price_range = st.sidebar.slider(
        "Price Range",
        int(df["Price"].min()),
        int(df["Price"].max()),
        (int(df["Price"].min()), int(df["Price"].max()))
    )

# ---------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------
filtered_df = df[
    (df["Location"].isin(locations)) &
    (df["Price"].between(price_range[0], price_range[1]))
]

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🏙️ AI Real Estate Copilot")

if demo_mode:
    st.markdown("<div class='spotlight'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='spotlight-box'>🎯 Spotlight Mode: Key insights highlighted</div>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# KPIs
# ---------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings", len(filtered_df))
c2.metric("Avg Price", f"₹ {int(filtered_df['Price'].mean()):,}")
c3.metric("Avg Area", f"{int(filtered_df['Total_Area'].mean())} sqft")
c4.metric("₹ / Sqft", f"₹ {int(filtered_df['Price_per_SQFT'].mean())}")

st.divider()

# ---------------------------------------------------------
# PRICE vs AREA (NEON HOVER)
# ---------------------------------------------------------
st.subheader("📊 Price vs Area")

fig = px.scatter(
    filtered_df,
    x="Total_Area",
    y="Price",
    color="Location",
    size="Price_per_SQFT",
    hover_name="Property Title",
    template="plotly_dark"
)
fig.update_layout(transition_duration=500)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# AI CHART NARRATION
# ---------------------------------------------------------
@st.cache_data
def narrate_chart(df):
    summary = df[["Total_Area", "Price", "Price_per_SQFT"]].describe().to_string()
    prompt = f"Explain key insights from this chart:\n{summary}"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120
    )
    return res.choices[0].message.content

with st.expander("🧠 Auto Chart Narration"):
    st.markdown(narrate_chart(filtered_df))

# ---------------------------------------------------------
# DATA TABLE
# ---------------------------------------------------------
st.divider()
st.subheader("📋 Property Explorer")
st.dataframe(filtered_df, use_container_width=True)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.caption("AI Real Estate Copilot • Neon UI • Spotlight Demo Mode")
