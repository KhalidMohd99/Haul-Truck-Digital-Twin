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

# Title
st.title("🚛 Mansourah-Massarah Mine: Haul Fleet Digital Twin")
st.markdown("Real-time monitoring and predictive maintenance system for autonomous haulage.")

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

# Sidebar Controls
st.sidebar.header("Control Panel")

# Map Settings
st.sidebar.subheader("Map Settings")
mapbox_token = st.sidebar.text_input("Mapbox Token (Optional)", type="password", help="Enter token for Satellite view. Leave empty for OpenStreetMap.")

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
selected_truck = st.sidebar.selectbox("Select Truck Detail", truck_ids, index=6) # Default to 7 (index 6)

# Main Dashboard Layout
col1, col2 = st.columns([2, 1])

# Filter data based on slider
current_status = get_fleet_status(df, selected_time)

with col1:
    st.subheader("📍 Real-Time Fleet Map (GPS)")
    map_fig = create_map_figure(current_status, routes, mapbox_token)
    st.plotly_chart(map_fig, use_container_width=True)

with col2:
    st.subheader("📊 Fleet Statistics")
    
    # KPI Metrics
    total_trucks = len(current_status)
    active_trucks = len(current_status[current_status['operational_state'] != 'QUEUE'])
    avg_speed = current_status['speed'].mean()
    
    kpi1, kpi2 = st.columns(2)
    kpi1.metric("Active Trucks", f"{active_trucks}/{total_trucks}")
    kpi2.metric("Avg Speed", f"{avg_speed:.1f} km/h")
    
    # State Distribution
    st.markdown("### Operational States")
    state_counts = current_status['operational_state'].value_counts()
    st.bar_chart(state_counts)
    
    # Alert Box
    st.markdown("### ⚠️ Health Alerts")
    critical_trucks = current_status[current_status['health_status'].isin(['WARNING', 'CRITICAL'])]
    
    if not critical_trucks.empty:
        for _, row in critical_trucks.iterrows():
            st.error(f"**Truck {row['truck_id']}**: {row['health_status']} - Vib: {row['vibration']:.2f}g | RUL: {row['rul_hours']:.1f}h")
    else:
        st.success("All systems nominal.")

# Truck Detail Section (Full Width)
st.markdown("---")
st.header(f"🔍 Telemetry Analysis: Truck {selected_truck}")

# Filter for selected truck only up to selected time
truck_history = df[(df['truck_id'] == selected_truck) & (df['timestamp'] <= selected_time)]

tab1, tab2, tab3 = st.tabs(["📉 Vibration Analysis", "🔥 Thermal Profile", "🛞 Tire Pressure"])

with tab1:
    st.subheader("Vibration & Bearing Health")
    vib_fig = create_sensor_plot(truck_history, selected_truck, 'vibration', "Vibration (g) - Primary Failure Indicator")
    st.plotly_chart(vib_fig, use_container_width=True)
    
    st.info("**Insight**: Vibration increasing > 0.6g indicates early bearing wear. Spikes > 0.8g suggest critical failure.")

with tab2:
    st.subheader("Engine Temperature")
    temp_fig = create_sensor_plot(truck_history, selected_truck, 'engine_temperature', "Engine Temperature (°C)")
    st.plotly_chart(temp_fig, use_container_width=True)

with tab3:
    st.subheader("Tire Pressure Monitoring")
    tire_fig = create_sensor_plot(truck_history, selected_truck, 'tire_pressure', "Tire Pressure (psi)")
    st.plotly_chart(tire_fig, use_container_width=True)
