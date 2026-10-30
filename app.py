import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Housing Market Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. HELPER FUNCTIONS ---
def hex_to_rgba(hex_color, alpha=0.2):
    if hex_color is None: return "rgba(0,0,0,0)"
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'
    return hex_color

# --- 3. THEME ENGINE ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'

def get_theme_constants():
    if st.session_state.theme == 'Dark':
        return {
            "bg": "radial-gradient(circle at 10% 20%, #0f172a 0%, #020617 100%)",
            "card_bg": "rgba(30, 41, 59, 0.5)",
            "text": "#ffffff",
            "accent": "#38bdf8",
            "border": "rgba(56, 189, 248, 0.2)",
            "chart_template": "plotly_dark",
            "sidebar": "rgba(15, 23, 42, 0.95)",
            "success": "#4ade80",
            "danger": "#f87171",
            "sankey_nodes": ["#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#fb7185"]
        }
    else:
        return {
            "bg": "linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)",
            "card_bg": "rgba(255, 255, 255, 0.9)",
            "text": "#0f172a",
            "accent": "#0284c7",
            "border": "rgba(2, 132, 199, 0.1)",
            "chart_template": "plotly_white",
            "sidebar": "rgba(255, 255, 255, 0.9)",
            "success": "#16a34a",
            "danger": "#dc2626",
            "sankey_nodes": ["#0284c7", "#6366f1", "#8b5cf6", "#d946ef", "#f43f5e"]
        }

theme = get_theme_constants()

# Inject CSS
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Rajdhani:wght@400;600&display=swap');
        
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        @keyframes pulse {{ 0% {{ transform: scale(1); }} 50% {{ transform: scale(1.02); }} 100% {{ transform: scale(1); }} }}
        @keyframes shimmer {{ 0% {{ background-position: -1000px 0; }} 100% {{ background-position: 1000px 0; }} }}

        html, body, [class*="css"], .stMarkdown, .stText, p, div, span, label, li, h1, h2, h3, h4, h5, h6 {{
            color: {theme['text']} !important;
            font-family: 'Rajdhani', sans-serif;
        }}
        
        .holo-title {{
            color: transparent !important;
            -webkit-text-fill-color: transparent !important;
        }}

        .stApp {{
            background: {theme['bg']} !important;
        }}
        
        h1, h2, h3, h4 {{ 
            font-family: 'Orbitron', sans-serif !important; 
            letter-spacing: 1px; 
        }}
        
        .holo-title {{
            background: linear-gradient(to right, {theme['accent']}, #818cf8, {theme['success']}, {theme['accent']});
            background-size: 200% auto;
            background-clip: text;
            -webkit-background-clip: text;
            animation: shimmer 5s linear infinite;
            font-weight: 800;
        }}
        
        .glass-card {{
            background: {theme['card_bg']};
            backdrop-filter: blur(12px);
            border: 1px solid {theme['border']};
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            animation: fadeIn 0.6s ease-out forwards;
        }}
        .glass-card:hover {{
            border-color: {theme['accent']};
            box-shadow: 0 15px 40px -10px {theme['accent']}40;
            transform: translateY(-5px);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }}
        
        /* TEAM MEMBER CARD */
        .team-card {{
            text-align: center;
            padding: 30px;
            border: 1px solid {theme['accent']};
            background: {hex_to_rgba(theme['card_bg'], 0.3)};
            border-radius: 20px;
            transition: transform 0.3s ease;
        }}
        .team-card:hover {{
            transform: scale(1.05);
            background: {hex_to_rgba(theme['accent'], 0.1)};
        }}
        
        /* AI INSIGHT BOX */
        .ai-insight {{
            background: {hex_to_rgba(theme['accent'], 0.1)};
            border-left: 4px solid {theme['accent']};
            padding: 10px 15px;
            border-radius: 0 8px 8px 0;
            margin-top: -10px;
            margin-bottom: 20px;
            font-size: 0.9rem;
            animation: fadeIn 1s ease-out;
        }}
        
        /* MOBILE SCROLLABLE TABS */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            white-space: nowrap;
            overflow-x: auto;
            flex-wrap: nowrap;
            scrollbar-width: thin;
            -webkit-overflow-scrolling: touch;
            padding-bottom: 5px;
        }}
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{
            height: 4px;
        }}
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {{
            background-color: {theme['accent']};
            border-radius: 4px;
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, {theme['accent']}, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family: 'Orbitron', sans-serif;
        }}
        
        [data-testid="stSidebar"] {{
            background-color: {theme['sidebar']} !important;
            border-right: 1px solid {theme['border']} !important;
        }}
        
        .stNumberInput input, .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{
            background-color: {theme['card_bg']} !important;
            color: {theme['text']} !important;
            -webkit-text-fill-color: {theme['text']} !important;
            border-color: {theme['border']} !important;
        }}
        
        .stSelectbox div[data-baseweb="select"] span {{
            color: {theme['text']} !important;
        }}
        
        .stButton>button {{
            background: linear-gradient(90deg, {theme['accent']}, #2563eb);
            color: white !important;
            border: none;
            font-family: 'Orbitron', sans-serif;
            transition: transform 0.2s;
        }}
        .stButton>button:hover {{
            transform: scale(1.05);
            box-shadow: 0 0 15px {theme['accent']};
        }}
        
        .badge {{
            background: {theme['accent']}20;
            color: {theme['accent']} !important;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
            animation: pulse 2s infinite;
        }}
        
        .vs-divider {{
            text-align: center; 
            font-size: 2.5rem; 
            font-weight: bold; 
            color: {theme['accent']};
            margin-top: 40px;
            font-family: 'Orbitron';
        }}
    </style>
""", unsafe_allow_html=True)

# --- 4. DATA LOADER ---
@st.cache_data(show_spinner=False)
def load_data():
    files = ['Real_Estate_Cleaned.xlsx', 'Real_Estate_Cleaned.xlsx - Sheet1.csv', 'Real_Estate_Cleaned.csv']
    df = pd.DataFrame()
    for f in files:
        try:
            if f.endswith('.xlsx'): df = pd.read_excel(f, engine='openpyxl')
            else: df = pd.read_csv(f)
            cols_lower = [c.lower() for c in df.columns]
            if not all(req in cols_lower for req in ['price', 'total_area', 'city']): continue
            break
        except Exception: continue
            
    if not df.empty:
        df = df.rename(columns={
            'Name': 'Project', 'Total_Area': 'Area', 'price': 'Price',
            'Price_per_SQFT': 'Rate_SqFt', 'city': 'City', 'suburb': 'Location',
            'Bed Rooms': 'BHK', 'Property Type': 'Type', 'Description': 'Description'
        })
        
        df['Price_Lakh'] = pd.to_numeric(df['Price'], errors='coerce') / 100000
        df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
        df['Rate_SqFt'] = pd.to_numeric(df['Rate_SqFt'], errors='coerce')
        df = df.dropna(subset=['Price_Lakh', 'Area', 'City'])
        df = df[(df['Area'] > 100) & (df['Price_Lakh'] > 5)]
        df['Location'] = df['Location'].fillna('Unknown')
        df['Project'] = df['Project'].fillna('Unknown Project')
        
        if 'Description' in df.columns:
            df['Description'] = df['Description'].astype(str).str.lower()
            df['Has_Gym'] = df['Description'].str.contains('gym|fitness', case=False)
            df['Has_Pool'] = df['Description'].str.contains('pool|swimming', case=False)
            df['Has_Metro'] = df['Description'].str.contains('metro|station', case=False)
        else:
            df['Has_Gym'] = False; df['Has_Pool'] = False; df['Has_Metro'] = False

        city_means = df.groupby('City')['Rate_SqFt'].transform('mean')
        df['Value_Delta'] = ((city_means - df['Rate_SqFt']) / city_means) * 100
        df['Sentiment'] = np.select(
            [df['Value_Delta'] > 15, df['Value_Delta'] < -15], 
            ['🟢 Undervalued', '🔴 Overpriced'], default='🟡 Fair Value'
        )
        
    return df

with st.spinner("🚀 Booting Up Housing Intelligence..."):
    df = load_data()

# --- 5. HYBRID AI ENGINE ---
class HybridPredictor:
    def __init__(self, data):
        self.use_ml = False
        self.data = data
        self.location_map = data.groupby('Location')['Rate_SqFt'].mean().to_dict()
        self.city_map = data.groupby('City')['Rate_SqFt'].mean().to_dict()
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder
            
            ml_df = data[['City', 'Location', 'Area', 'BHK', 'Price_Lakh']].dropna()
            self.le_city = LabelEncoder()
            ml_df['City_Code'] = self.le_city.fit_transform(ml_df['City'])
            ml_df['Loc_Score'] = ml_df['Location'].map(self.location_map).fillna(0)
            
            X = ml_df[['City_Code', 'Loc_Score', 'Area', 'BHK']]
            y = ml_df['Price_Lakh']
            
            self.model = RandomForestRegressor(n_estimators=20, max_depth=8, n_jobs=-1)
            self.model.fit(X, y)
            self.use_ml = True
        except ImportError:
            self.use_ml = False
            
    def predict(self, city, location, area, bhk):
        if self.use_ml:
            try:
                c_code = self.le_city.transform([city])[0]
                l_score = self.location_map.get(location, self.city_map.get(city, 0))
                return self.model.predict([[c_code, l_score, area, bhk]])[0]
            except:
                return self.heuristic_predict(city, location, area)
        else:
            return self.heuristic_predict(city, location, area)
            
    def heuristic_predict(self, city, location, area):
        rate = self.location_map.get(location, self.city_map.get(city, 5000))
        return (rate * area) / 100000

if not df.empty:
    predictor = HybridPredictor(df)

# --- 6. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("## 🧬 DASHBOARD CONTROL")
    if st.button("🌗 Toggle Light/Dark Theme"):
        st.session_state.theme = 'Light' if st.session_state.theme == 'Dark' else 'Dark'
        st.rerun()
    st.markdown("---")
    
    with st.expander("🔮 AI Price Oracle", expanded=False):
        st.caption("Estimate Fair Market Value instantly.")
        if not df.empty:
            p_city = st.selectbox("🏙️ City", sorted(df['City'].unique()))
            city_locs = sorted(df[df['City'] == p_city]['Location'].unique())
            p_loc = st.selectbox("📍 Location", city_locs) if city_locs else "Unknown"
            p_area = st.number_input("📐 Area (SqFt)", 500, 10000, 1200)
            p_bhk = st.slider("🛏️ BHK", 1, 6, 2)
            
            if st.button("🚀 Predict Value"):
                val = predictor.predict(p_city, p_loc, p_area, p_bhk)
                st.markdown(f"""
                    <div style="background:{theme['accent']}20; padding:15px; border-radius:10px; border:1px solid {theme['accent']}; margin-top:10px;">
                        <div style="font-size:12px;">🤖 AI ESTIMATE</div>
                        <div style="font-size:24px; font-weight:bold; color:{theme['accent']}; font-family:'Orbitron'">₹ {val:.1f} L</div>
                        <div style="font-size:10px; opacity:0.7;">{'⚡ Neural Net Prediction' if predictor.use_ml else '📊 Market Rate Logic'}</div>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown("### 🧘 Lifestyle Filters")
    amenities = []
    if st.checkbox("💪 Gym / Fitness"): amenities.append('Has_Gym')
    if st.checkbox("🏊 Pool"): amenities.append('Has_Pool')
    if st.checkbox("🚇 Metro Nearby"): amenities.append('Has_Metro')

    st.markdown("---")

    if not df.empty:
        all_cities = sorted(df['City'].unique())
        cities = st.multiselect("🏙️ Filter City", all_cities, default=all_cities)
        bhk_rng = st.slider("🛏️ Filter BHK", 1, 6, (1, 5))
        price_rng = st.slider("💰 Filter Budget (₹ L)", 10, 1000, (10, 500))
        
        mask = (df['City'].isin(cities)) & (df['BHK'].between(*bhk_rng)) & (df['Price_Lakh'].between(*price_rng))
        for amen in amenities:
            mask = mask & (df[amen] == True)
        df_filt = df[mask]
    else:
        df_filt = pd.DataFrame()
    
    if not df_filt.empty:
        csv = df_filt.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Data", csv, "nexus_report.csv", "text/csv")

# --- 7. MAIN DASHBOARD ---

# HERO
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown(f"""
        <h1 style="font-size: 3.5rem; margin-bottom: 0;" class="holo-title">HOUSING MARKET <span style="font-size: 2rem;">///</span> INTELLIGENCE</h1>
    """, unsafe_allow_html=True)
    st.markdown(f'<p style="font-size: 1.2rem; opacity: 0.8;">🚀 Next-Gen Real Estate Intelligence Dashboard</p>', unsafe_allow_html=True)
with c2:
    if not df_filt.empty:
        st.markdown(f"""
            <div style="text-align: right; margin-top: 20px;">
                <span class="badge">📡 LIVE DATA FEED</span>
                <h2 style="margin:0; font-family:'Orbitron'">{len(df_filt):,}</h2>
                <span style="font-size:12px; opacity:0.6;">Active Properties</span>
            </div>
        """, unsafe_allow_html=True)

if not df_filt.empty:

    # METRICS
    k1, k2, k3, k4 = st.columns(4)
    avg_p = df_filt['Price_Lakh'].mean()
    avg_r = df_filt['Rate_SqFt'].mean()
    
    def metric_card(col, label, val, sub, icon, delay):
        col.markdown(f"""
            <div class="glass-card" style="animation-delay: {delay}s">
                <div style="font-size:12px; letter-spacing:1px; opacity:0.7; text-transform:uppercase;">{icon} {label}</div>
                <div class="metric-value">{val}</div>
                <div style="font-size:11px; opacity:0.6;">{sub}</div>
            </div>
        """, unsafe_allow_html=True)
        
    metric_card(k1, "Market Avg", f"₹{avg_p:.0f}L", "Base Valuation", "🏷️", "0.1")
    metric_card(k2, "Rate Density", f"₹{avg_r:,.0f}", "Per Sq. Ft.", "📏", "0.2")
    metric_card(k3, "Opportunities", f"{len(df_filt[df_filt['Sentiment']=='🟢 Undervalued'])}", "Undervalued Units", "💎", "0.3")
    metric_card(k4, "Inventory", f"{len(df_filt)}", "Units Found", "📦", "0.4")

    # TABS
    t1, t2, t3, t4, t5, t6 = st.tabs(["🧬 MARKET DNA", "⚔️ VERSUS BATTLE", "🤖 AI VERDICT", "⚖️ WEALTH ENGINE", "🗃️ DATA GRID", "👥 TEAM"])

    # TAB 1: CHARTS
    with t1:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown("### 🌈 Market Flow (City → Type)")
            sankey_data = df_filt.groupby(['City', 'Type'])['Price'].count().reset_index()
            sankey_data.columns = ['Source', 'Target', 'Value']
            
            all_labels = list(sankey_data['Source'].unique()) + list(sankey_data['Target'].unique())
            palette = theme['sankey_nodes']
            node_colors = [palette[i % len(palette)] for i in range(len(all_labels))]
            link_opacity = "0.3" if st.session_state.theme == 'Dark' else "0.2"
            
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_labels, color=node_colors),
                link=dict(source=[all_labels.index(x) for x in sankey_data['Source']], target=[all_labels.index(x) for x in sankey_data['Target']], value=sankey_data['Value'], color=theme['accent'].replace(')', f', {link_opacity})').replace('rgb', 'rgba'))
            )])
            fig_sankey.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Rajdhani", size=12, color=theme['text']), margin=dict(t=20, l=10, r=10, b=10))
            st.plotly_chart(fig_sankey, use_container_width=True)
            
            # AI Insight for Sankey
            try:
                top_source = sankey_data.groupby('Source')['Value'].sum().idxmax()
                top_target = sankey_data.groupby('Target')['Value'].sum().idxmax()
                st.markdown(f"""<div class="ai-insight">🤖 <b>AI Analysis:</b> The market data indicates that <b>{top_source}</b> is the dominant supply hub, with a massive consumer preference shifting towards <b>{top_target}</b> properties.</div>""", unsafe_allow_html=True)
            except: pass

        with c_right:
            st.markdown("### 🕸️ Asset Hologram")
            radar_cols = ['Price_Lakh', 'Rate_SqFt', 'Area']
            radar_data = df_filt.groupby('City')[radar_cols].mean()
            if not radar_data.empty:
                radar_data = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())
                radar_data = radar_data.reset_index().melt(id_vars='City')
                fig_radar = px.line_polar(radar_data, r='value', theta='variable', color='City', line_close=True, color_discrete_sequence=px.colors.sequential.Teal)
                fig_radar.update_layout(template=theme['chart_template'], paper_bgcolor='rgba(0,0,0,0)', polar=dict(bgcolor="rgba(0,0,0,0.05)", radialaxis=dict(showticklabels=False)), margin=dict(t=20, b=20), legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # AI Insight for Radar
                st.markdown(f"""<div class="ai-insight">🤖 <b>AI Analysis:</b> Multi-dimensional scanning suggests varying value propositions across cities, with distinct trade-offs between <b>Area</b> and <b>Price Density</b>.</div>""", unsafe_allow_html=True)

        st.markdown("---")
        c_scat, c_pie = st.columns([2, 1])
        
        with c_scat:
            st.markdown("### 🔵 Price vs. Area Matrix")
            fig_scatter = px.scatter(
                df_filt, x='Area', y='Price_Lakh', 
                color='BHK', size='Price_Lakh',
                hover_data=['Project', 'Location'],
                color_discrete_sequence=px.colors.qualitative.Prism,
            )
            fig_scatter.update_layout(
                template=theme['chart_template'],
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0.05)',
                font=dict(family="Rajdhani", color=theme['text']),
                xaxis=dict(gridcolor=theme['border'], title="Area (SqFt)"),
                yaxis=dict(gridcolor=theme['border'], title="Price (Lakhs)")
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # AI Insight for Scatter
            corr = df_filt['Area'].corr(df_filt['Price_Lakh'])
            trend_txt = "strong positive" if corr > 0.7 else "moderate"
            st.markdown(f"""<div class="ai-insight">🤖 <b>AI Analysis:</b> Detected a <b>{trend_txt} correlation ({corr:.2f})</b>. Larger floor plans are commanding disproportionately higher premiums in this filtered segment.</div>""", unsafe_allow_html=True)
            
        with c_pie:
            st.markdown("### 🍩 BHK Distribution")
            fig_donut = px.pie(
                df_filt, names='BHK', 
                hole=0.6,
                color_discrete_sequence=px.colors.qualitative.T10
            )
            fig_donut.add_annotation(
                text=f"{len(df_filt)}<br>Units", 
                x=0.5, y=0.5, font_size=20, showarrow=False, 
                font_family="Orbitron", font_color=theme['text']
            )
            fig_donut.update_traces(textposition='outside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
            fig_donut.update_layout(
                template=theme['chart_template'],
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Rajdhani", color=theme['text']),
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
            # AI Insight for Donut
            try:
                top_bhk = df_filt['BHK'].mode()[0]
                pct = (len(df_filt[df_filt['BHK']==top_bhk])/len(df_filt))*100
                st.markdown(f"""<div class="ai-insight">🤖 <b>AI Analysis:</b> <b>{top_bhk} BHK</b> units are the dominant asset class, comprising <b>{pct:.1f}%</b> of the visible inventory.</div>""", unsafe_allow_html=True)
            except: pass

        st.markdown("### 📊 Average Cost per BHK")
        bhk_price = df_filt.groupby('BHK')['Price_Lakh'].mean().reset_index()
        fig_bar = px.bar(
            bhk_price, x='BHK', y='Price_Lakh',
            color='Price_Lakh',
            color_continuous_scale='Plasma',
            text_auto='.1f'
        )
        fig_bar.update_layout(
            template=theme['chart_template'],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Rajdhani", color=theme['text']),
            xaxis=dict(tickmode='linear', title="Bedroom Configuration", type='category'),
            yaxis=dict(showgrid=False, title="Avg Price (Lakhs)"),
            coloraxis_showscale=False,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # AI Insight for Bar
        try:
            min_p = bhk_price['Price_Lakh'].min()
            max_p = bhk_price['Price_Lakh'].max()
            st.markdown(f"""<div class="ai-insight">🤖 <b>AI Analysis:</b> The price ceiling for luxury configurations peaks at <b>₹{max_p:.1f}L</b>, representing a significant multiplier over the entry-level average of <b>₹{min_p:.1f}L</b>.</div>""", unsafe_allow_html=True)
        except: pass

    # TAB 2: VERSUS MODE
    with t2:
        st.markdown("### ⚔️ Project Face-off")
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            proj_a = st.selectbox("🥊 Select Challenger A", df_filt['Project'].unique(), key="p_a")
        with c_sel2:
            remaining = [p for p in df_filt['Project'].unique() if p != proj_a]
            proj_b = st.selectbox("🥊 Select Challenger B", remaining, key="p_b") if remaining else None
        
        if proj_a and proj_b:
            row_a = df_filt[df_filt['Project'] == proj_a].iloc[0]
            row_b = df_filt[df_filt['Project'] == proj_b].iloc[0]
            
            c_a, c_mid, c_b = st.columns([1, 0.2, 1])
            with c_a:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center; border-color:{theme['accent']};">
                    <h2>{proj_a}</h2>
                    <h1 style="color:{theme['accent']}">₹ {row_a['Price_Lakh']:.1f} L</h1>
                    <p>📏 {row_a['Area']} sqft | 🛏️ {row_a['BHK']} BHK</p>
                    <p style="color:{theme['accent']}">🏷️ ₹ {row_a['Rate_SqFt']:,.0f} / sqft</p>
                    <span class="badge">📍 {row_a['Location']}</span>
                </div>
                """, unsafe_allow_html=True)
            with c_mid:
                st.markdown('<div class="vs-divider">⚡ VS ⚡</div>', unsafe_allow_html=True)
            with c_b:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center; border-color:{theme['danger']};">
                    <h2>{proj_b}</h2>
                    <h1 style="color:{theme['danger']}">₹ {row_b['Price_Lakh']:.1f} L</h1>
                    <p>📏 {row_b['Area']} sqft | 🛏️ {row_b['BHK']} BHK</p>
                    <p style="color:{theme['danger']}">🏷️ ₹ {row_b['Rate_SqFt']:,.0f} / sqft</p>
                    <span class="badge">📍 {row_b['Location']}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 📊 Spec Comparison")
            categories = ['Price (L)', 'Area (sqft)', 'Rate (sqft)']
            fig_vs = go.Figure()
            max_p = max(row_a['Price_Lakh'], row_b['Price_Lakh'])
            max_a = max(row_a['Area'], row_b['Area'])
            max_r = max(row_a['Rate_SqFt'], row_b['Rate_SqFt'])
            vals_a = [row_a['Price_Lakh']/max_p, row_a['Area']/max_a, row_a['Rate_SqFt']/max_r]
            vals_b = [row_b['Price_Lakh']/max_p, row_b['Area']/max_a, row_b['Rate_SqFt']/max_r]
            fig_vs.add_trace(go.Scatterpolar(r=vals_a, theta=categories, fill='toself', name=proj_a, line_color=theme['accent']))
            fig_vs.add_trace(go.Scatterpolar(r=vals_b, theta=categories, fill='toself', name=proj_b, line_color=theme['danger']))
            fig_vs.update_layout(template=theme['chart_template'], paper_bgcolor='rgba(0,0,0,0)', polar=dict(radialaxis=dict(visible=False)))
            st.plotly_chart(fig_vs, use_container_width=True)
            
            # AI Insight for Versus
            cheaper = proj_a if row_a['Price_Lakh'] < row_b['Price_Lakh'] else proj_b
            bigger = proj_a if row_a['Area'] > row_b['Area'] else proj_b
            st.markdown(f"""<div class="ai-insight">🤖 <b>AI Analysis:</b> In this head-to-head, <b>{cheaper}</b> wins on affordability, whereas <b>{bigger}</b> offers superior living space. The choice depends on your price-per-sqft sensitivity.</div>""", unsafe_allow_html=True)

    # TAB 3: AI VERDICT
    with t3:
        st.markdown("### 🤖 Intelligent Project Analysis")
        target_proj = st.selectbox("🔍 Analyze Project", df_filt['Project'].unique())
        if target_proj:
            row = df_filt[df_filt['Project'] == target_proj].iloc[0]
            loc_avg = df[df['Location'] == row['Location']]['Rate_SqFt'].mean()
            diff = ((row['Rate_SqFt'] - loc_avg) / loc_avg) * 100
            verdict = "STRONG BUY 🚀" if diff < -10 else "FAIR VALUE ⚖️" if diff < 10 else "PREMIUM PRICED 💎"
            color = theme['success'] if diff < -10 else theme['accent'] if diff < 10 else theme['danger']
            c_v1, c_v2 = st.columns([1, 2])
            with c_v1:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center;">
                    <h3 style="color:{color}">VERDICT</h3>
                    <h1 style="color:{color}">{verdict}</h1>
                </div>
                """, unsafe_allow_html=True)
            with c_v2:
                st.markdown(f"""
                <div class="glass-card">
                    <h4>📝 Neural Summary</h4>
                    <ul>
                        <li>This property in <b>{row['Location']}</b> is priced <b>{abs(diff):.1f}% {'below' if diff < 0 else 'above'}</b> the neighborhood average.</li>
                        <li>It offers <b>{row['Area']} sqft</b> at <b>₹{row['Rate_SqFt']:,.0f}/sqft</b>.</li>
                        <li>Based on current market trends, this asset shows <b>{'high 📈' if diff < 0 else 'moderate 📊'} appreciation potential</b>.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # TAB 4: WEALTH & TOOLS
    with t4:
        st.markdown("### ⚖️ Financial Toolkit")
        c_calc1, c_calc2 = st.columns(2)
        with c_calc1:
            st.markdown(f"""<div class="glass-card"><h3>📈 Wealth Projector (20 Years)</h3><p style="opacity:0.7;">Rent vs. Buy Simulation</p></div>""", unsafe_allow_html=True)
            p_val_proj = st.number_input("💵 Property Value (₹ L)", 20, 1000, 75, key="proj_val")
            appreciation = st.slider("📈 Annual Appreciation (%)", 1, 15, 5)
            years = list(range(1, 21))
            values = [p_val_proj * ((1 + appreciation/100)**y) for y in years]
            fig_wealth = px.area(x=years, y=values, title="Projected Asset Value")
            fill_rgba = hex_to_rgba(theme['success'], 0.2)
            fig_wealth.update_traces(line_color=theme['success'], fillcolor=fill_rgba)
            fig_wealth.update_layout(template=theme['chart_template'], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_wealth, use_container_width=True)
            
            # AI Insight for Wealth
            final_gain = values[-1] - values[0]
            st.markdown(f"""<div class="ai-insight">🤖 <b>AI Analysis:</b> At this trajectory, the asset is projected to generate a capital gain of <b>₹{final_gain:.1f}L</b> over two decades, assuming market stability.</div>""", unsafe_allow_html=True)
            
        with c_calc2:
            st.markdown(f"""<div class="glass-card"><h3>🧾 TruePrice™ Estimator</h3><p style="opacity:0.7;">Inclusive of Taxes</p></div>""", unsafe_allow_html=True)
            base_p = st.number_input("💵 Base Price (₹ L)", 20, 1000, 75, key="true_price")
            city_tier = st.selectbox("🏙️ City Tier", ["Tier 1 (Metros)", "Tier 2"])
            stamp = 0.06 if "Tier 1" in city_tier else 0.07
            final = base_p * (1 + stamp + 0.05 + 0.01)
            st.markdown(f"""<h2 style='color:{theme['accent']}'>₹ {final:.2f} L</h2>""", unsafe_allow_html=True)
            with st.expander("ℹ️ Breakdown"): st.markdown(f"**Stamp Duty:** {stamp*100}%\n**GST:** 5%\n**Registration:** 1%")

    # TAB 5: DATA GRID
    with t5:
        st.markdown("### 🗃️ Raw Data Explorer")
        st.dataframe(df_filt, use_container_width=True, height=500)

    # TAB 6: TEAM
    with t6:
        st.markdown(f"<h2 style='text-align: left; color: {theme['accent']}'>GROUP NO. 4</h2>", unsafe_allow_html=True)
        holo_mode = 50
            
        members = [
            {"name": "Aman Singh", "role": "Dashboard/UI/ML"},
            {"name": "Shubham Missar", "role": "Data Architect"},
            {"name": "Soumya Mathur", "role": "Data Research"},
            {"name": "Navdeep", "role": "Data Research"},
            {"name": "Ishtehaq", "role": "Data Research"}
        ]
        
        # Holo Effect CSS injection
        st.markdown(f"""
        <style>
            .team-card {{
                box-shadow: 0 0 {holo_mode}px {theme['accent']} !important;
                border-color: {hex_to_rgba(theme['accent'], holo_mode/100)} !important;
            }}
        </style>
        """, unsafe_allow_html=True)

        cols = st.columns(3)
        for i, member in enumerate(members):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="glass-card team-card">
                    <div style="font-size: 40px; margin-bottom: 10px;">👨‍💻</div>
                    <h3 style="margin: 0; color: {theme['text']}">{member['name']}</h3>
                    <p style="color: {theme['accent']}; font-size: 14px; letter-spacing: 1px;">{member['role']}</p>
                </div>
                """, unsafe_allow_html=True)

else:
    st.error("⚠️ NEXUS OFFLINE: Data Source Not Found.")
