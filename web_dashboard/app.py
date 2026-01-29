import streamlit as st
import pandas as pd
from datetime import timedelta
import plotly.express as px
from utils import load_data, load_routes, get_fleet_status, create_map_figure, create_sensor_plot

# Page Configuration
st.set_page_config(
    page_title="Haul Truck Digital Twin",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Control Room" Feel
st.markdown("""
<style>
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4e4e4e;
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🚛 Mansourah-Massarah: Mine Operations Command Center")
st.markdown("Real-time digital twin for autonomous haulage optimization.")

# Load Data
@st.cache_data
def get_dataset():
    return load_data()

@st.cache_data
def get_routes():
    return load_routes()

df = get_dataset()
routes = get_routes()

if df.empty:
    st.error("Data not found! Please check 'data/GPS Enhancement/mansourah_haul_truck_telemetry_with_gps.csv'")
    st.stop()

# --- TOP KPI RIBBON ---
st.markdown("### 📊 Operational KPIs")
k1, k2, k3, k4, k5 = st.columns(5)

# Calculate KPIs (Simulated for Demo)
# 1. Fleet Availability
availability_pct = 95.2 
k1.metric("Fleet Availability", f"{availability_pct}%", "+1.3%")

# 2. Trucks at Risk
trucks_at_risk = df[df['risk_level'] == 'CRITICAL']['truck_id'].nunique()
# Using generic logic for delta - assuming 1 less than yesterday
k2.metric("Trucks at Risk (48h)", f"{trucks_at_risk} Units", "-1", delta_color="inverse")

# 3. Avg RUL
avg_rul = df['rul_hours'].mean()
k3.metric("Avg RUL", f"{avg_rul:,.0f} Hours", "-12h")

# 4. MTBF
k4.metric("MTBF", "482 Hours", "+5.4%")

# 5. Cost Avoided
k5.metric("Est. Cost Avoided", "$1.2M", "+$50k")

st.markdown("---")

# Sidebar Controls
st.sidebar.header("🕹️ Control Panel")

# Map Settings
st.sidebar.subheader("Map Settings")
mapbox_token = st.sidebar.text_input("Mapbox Token (Optional)", type="password", help="Enter token for Satellite view. Leave empty for High-Contrast Dark Mode.")

# Time Simulation Slider
min_time = df['timestamp'].min()
max_time = df['timestamp'].max()

st.sidebar.subheader("Replay Simulation")
selected_time = st.sidebar.slider(
    "Current Time",
    min_value=min_time.to_pydatetime(),
    max_value=max_time.to_pydatetime(),
    value=max_time.to_pydatetime(),
    step=timedelta(minutes=30),
    format="MM-DD HH:mm"
)

# Truck Selection
truck_ids = sorted(df['truck_id'].unique())
selected_truck = st.sidebar.selectbox("Select Asset for Deep Dive", truck_ids, index=6) # Default to 7 (index 6)

# Main Dashboard Layout
col1, col2 = st.columns([2, 1])

# Filter data based on slider
current_status = get_fleet_status(df, selected_time)

with col1:
    st.subheader("🗺️ Operational Intelligence Map")
    # Pass all data for trails
    map_fig = create_map_figure(current_status, df, selected_time, routes, mapbox_token)
    st.plotly_chart(map_fig, use_container_width=True)

with col2:
    st.subheader("🛠️ Maintenance Decision Simulator")
    st.info("💡 **Scenario Analysis**: Select an action to see impact.")
    
    action = st.radio("Action:", ["✅ Perform Maintenance Now", "❌ Delay Maintenance (Risk)"])
    
    if "Delay" in action:
        st.error("⚠️ **PROJECTED RISK INCREASE**")
        c1, c2 = st.columns(2)
        c1.metric("Risk Factor", "CRITICAL", "High Prop.")
        c2.metric("Est. Downtime Cost", "$450k", "+$300k", delta_color="inverse")
        st.markdown("**Projected Failure:** < 24 Hours")
    else:
        st.success("🛡️ **OPTIMAL STRATEGY**")
        c1, c2 = st.columns(2)
        c1.metric("Risk Factor", "LOW", "Stable")
        c2.metric("Maintenance Cost", "$15k", "-$435k")
        st.markdown("**Outcome:** Asset Returned to Service in 4h")

    st.subheader("⚠️ Critical Alerts")
    critical_trucks = current_status[current_status['risk_level'].isin(['WARNING', 'CRITICAL'])]
    
    if not critical_trucks.empty:
        for _, row in critical_trucks.iterrows():
            severity = "🔴" if row['risk_level'] == 'CRITICAL' else "🟡"
            st.markdown(f"{severity} **Truck {row['truck_id']}**: RUL {row['rul_hours']:.1f}h | Vib {row['vibration']:.2f}g")
    else:
        st.success("All systems nominal.")

# Truck Detail Section (Full Width)
st.markdown("---")
st.header(f"🔍 Asset Health Intelligence: Truck {selected_truck}")

# Filter for selected truck only up to selected time
truck_history = df[(df['truck_id'] == selected_truck) & (df['timestamp'] <= selected_time)]

# Layout: Chart vs AI Insight
c_chart, c_insight = st.columns([3, 1])

with c_insight:
    st.markdown("### 🤖 AI Diagnosis")
    
    # Dynamic insight based on truck data
    latest_vib = truck_history['vibration'].iloc[-1]
    
    if latest_vib > 0.8:
        st.error("**CRITICAL: Bearing Failure Imminent**")
        st.markdown("""
        **Failure Progression:** Rapid Acceleration
        
        **Est. Failure Window:** 12-18 Hours
        
        **Root Cause:** Inner Race Spalling
        
        **Rec. Action:** Stop immediately. Inspect Hub 3.
        """)
    elif latest_vib > 0.6:
        st.warning("**WARNING: Early Degradation**")
        st.markdown("""
        **Failure Progression:** Gradual Drift
        
        **Est. Failure Window:** 48-72 Hours
        
        **Rec. Action:** Schedule for next PM window.
        """)
    else:
        st.success("**HEALTHY: Optimal Operation**")
        st.markdown("""
        **Status:** Nominal
        
        **Efficiency:** 98.4%
        
        **Next Service:** 250 Hours
        """)

with c_chart:
    tab1, tab2, tab3 = st.tabs(["📉 Primary Failure Signal", "🔥 Thermal Profile", "🛞 Tire Telemetry"])

    with tab1:
        st.subheader("Vibration Analysis (Bearing Health)")
        vib_fig = create_sensor_plot(truck_history, selected_truck, 'vibration', "Vibration (g)")
        st.plotly_chart(vib_fig, use_container_width=True)

    with tab2:
        st.subheader("Engine Temperature")
        temp_fig = create_sensor_plot(truck_history, selected_truck, 'engine_temperature', "Engine Temperature (°C)")
        st.plotly_chart(temp_fig, use_container_width=True)

    with tab3:
        st.subheader("Tire Pressure Monitoring")
        tire_fig = create_sensor_plot(truck_history, selected_truck, 'tire_pressure', "Tire Pressure (psi)")
        st.plotly_chart(tire_fig, use_container_width=True)

