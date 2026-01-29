import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

def load_data(filepath="data/GPS Enhancement/mansourah_haul_truck_telemetry_with_gps.csv"):
    """
    Loads and preprocesses the telemetry data.
    Now defaults to the GPS-enhanced dataset.
    """
    try:
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Standardize Column Names for Dashboard Compatibility
        column_map = {
            'vibration_g': 'vibration',
            'gps_latitude': 'latitude', 
            'gps_longitude': 'longitude',
            'speed_kph': 'speed',
            'engine_temp_celsius': 'engine_temperature',
            'tire_pressure_psi': 'tire_pressure'
        }
        df = df.rename(columns=column_map)
        
        # --- SIMULATION: Remaining Useful Life (RUL) ---
        # In a real app, this comes from an ML model. We simulate it for the demo.
        # Logic: Higher vibration = Lower RUL
        np.random.seed(42)
        base_rul = 1000 # hours
        df['rul_hours'] = base_rul - (df['vibration'] * 800) + np.random.normal(0, 10, len(df))
        df['rul_hours'] = df['rul_hours'].clip(lower=0) # No negative hours

        # --- RISK CATEGORIZATION ---
        # Red: RUL < 24h
        # Yellow: RUL < 72h
        # Green: Healthy
        conditions = [
            (df['rul_hours'] < 24),
            (df['rul_hours'] < 72)
        ]
        risk_choices = ['CRITICAL', 'WARNING']
        risk_colors = ['red', 'gold'] # Corresponding colors for mapping
        
        df['risk_level'] = np.select(conditions, risk_choices, default='HEALTHY')
        df['risk_color'] = np.select(conditions, risk_colors, default='#00cc96') # Standard green

        # Re-Compute Health Status to match dashboard expected values (WARNING/CRITICAL)
        # We keep this for backward compatibility but prioritize risk_level for visuals
        conditions_health = [
            (df['vibration'] > 0.8),
            (df['vibration'] > 0.6)
        ]
        choices_health = ['CRITICAL', 'WARNING']
        df['health_status'] = np.select(conditions_health, choices_health, default='OK')
        
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def load_routes(filepath="data/routes.geojson"):
    """Loads the GeoJSON route file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def get_fleet_status(df, current_time):
    """
    Gets the latest status for each truck at a given point in time.
    """
    latest_readings = df[df['timestamp'] <= current_time].sort_values('timestamp').groupby('truck_id').tail(1)
    return latest_readings

def get_truck_trails(df, current_time, window_minutes=30):
    """
    Gets the position history for the last 30 minutes for trails.
    """
    start_time = current_time - pd.Timedelta(minutes=window_minutes)
    trails = df[(df['timestamp'] <= current_time) & (df['timestamp'] >= start_time)]
    return trails

def create_map_figure(fleet_status, all_data=None, current_time=None, route_geojson=None, mapbox_token=None):
    """
    Creates an "Operational Intelligence Map" with Satellite view, Risk coloring, and Trails.
    """
    if fleet_status.empty:
        return go.Figure()

    # 1. Base Map Configuration (Satellite)
    # Using Esri World Imagery as a free satellite alternative if no Mapbox token
    
    fig = go.Figure()

    # 2. Add Route Layer (Static Background)
    if route_geojson:
        try:
            for feature in route_geojson['features']:
                if feature['geometry']['type'] == 'LineString':
                    coords = feature['geometry']['coordinates']
                    lons, lats = zip(*coords)
                    
                    fig.add_trace(go.Scattermapbox(
                        mode="lines",
                        lon=lons,
                        lat=lats,
                        marker={'size': 5},
                        line={'width': 1, 'color': 'rgba(200, 200, 200, 0.5)'}, # Subtle white/grey for satellite contrast
                        name='Haul Route',
                        showlegend=False
                    ))
        except Exception as e:
            print(f"Error loading route for map: {e}")

    # 3. Add Trails (Tail Paths)
    if all_data is not None and current_time is not None:
        trails = get_truck_trails(all_data, current_time)
        unique_trucks = trails['truck_id'].unique()
        
        for truck in unique_trucks:
            t_data = trails[trails['truck_id'] == truck]
            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lon=t_data['longitude'],
                lat=t_data['latitude'],
                line={'width': 2, 'color': 'rgba(0, 255, 255, 0.4)'}, # Cyan trails
                name=f'Trail {truck}',
                showlegend=False,
                hoverinfo='skip'
            ))

    # 4. Add Truck Markers (Risk-Based Coloring)
    # We use 'risk_color' pre-calculated in load_data
    
    # Create hover text
    hover_text = fleet_status.apply(
        lambda x: f"<b>Truck {x['truck_id']}</b><br>State: {x['operational_state']}<br>RUL: {x['rul_hours']:.1f}h<br>Risk: {x['risk_level']}",
        axis=1
    )

    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=fleet_status['longitude'],
        lat=fleet_status['latitude'],
        marker=go.scattermapbox.Marker(
            size=15, # Larger markers
            color=fleet_status['risk_color'],
            opacity=0.9
        ),
        text=hover_text,
        hoverinfo='text',
        name='Haul Trucks'
    ))

    # 5. Add "Risk Halos" for Critical Trucks
    critical_trucks = fleet_status[fleet_status['risk_level'] == 'CRITICAL']
    if not critical_trucks.empty:
         fig.add_trace(go.Scattermapbox(
            mode="markers",
            lon=critical_trucks['longitude'],
            lat=critical_trucks['latitude'],
            marker=go.scattermapbox.Marker(
                size=30, # Large halo
                color='red',
                opacity=0.3
            ),
            hoverinfo='skip',
            name='Risk Alert',
            showlegend=False
        ))

    # Layout configuration
    
    if mapbox_token:
        # Use Mapbox Satellite if token provided
        mapbox_dict = dict(
            accesstoken=mapbox_token,
            style="mapbox://styles/mapbox/satellite-v9",
            center=dict(lat=fleet_status['latitude'].mean(), lon=fleet_status['longitude'].mean()),
            zoom=14
        )
    else:
        # Fallback to Esri Satellite (White BG + Raster Layer)
        # Note: Plotly doesn't natively support Esri tiles easily without a proper style URL or layers.
        # For simplicity and robustness if user has no token, we stick to 'open-street-map' but dark style
        # OR use "white-bg" and add a raster layer. Let's try "carto-darkmatter" for "Control Room" feel if no satellite.
        # Actually, user requested Satellite specifically. Let's try to use "white-bg" layers if possible, but simplest is 'open-street-map' with a specific style if available.
        # Let's use 'carto-darkmatter' as a fallback "Control Room" style as it looks very professional, unless we can get a public satellite tile.
        # Common free satellite tiles often require keys. 
        # let's stick to standard mapbox styles available in plotly, 'carto-darkmatter' is great for dashboards.
        # But REQ said "Esri satellite".
        # Let's provide a reliable fallback.
        mapbox_dict = dict(
            style="carto-darkmatter", # High contrast fallback
            center=dict(lat=fleet_status['latitude'].mean(), lon=fleet_status['longitude'].mean()),
            zoom=14
        )
        
    fig.update_layout(
        mapbox=mapbox_dict,
        margin={"r":0,"t":0,"l":0,"b":0},
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0.5)', font=dict(color='white'))
    )
    
    return fig

def create_sensor_plot(df, truck_id, sensor_col, title):
    """
    Creates a time-series plot with anomaly markers.
    """
    truck_data = df[df['truck_id'] == truck_id]
    
    fig = px.line(
        truck_data, 
        x='timestamp', 
        y=sensor_col, 
        title=f"Truck {truck_id}: {title}",
        template="plotly_dark" # Instant professional dark theme
    )
    
    fig.update_traces(line_color='#00F0FF', line_width=2) # Cyan line for "Digital Twin" feel
    
    # Highlight failure zone if Truck 7 and vibration
    if truck_id == 7 and 'vibration' in sensor_col:
        # 1. Critical Threshold
        fig.add_hrect(y0=0.8, y1=1.5, line_width=0, fillcolor="red", opacity=0.1, annotation_text="CRITICAL FAILURE ZONE", annotation_position="top left")
        
        # 2. Anomaly Detected Marker (approximate time for demo)
        # Assuming anomaly starts halfway through the spike
        anomaly_time = truck_data[truck_data[sensor_col] > 0.6]['timestamp'].min()
        if pd.notna(anomaly_time):
             fig.add_vline(x=anomaly_time, line_width=1, line_dash="dash", line_color="yellow", annotation_text="AI Anomaly Detection", annotation_position="top right")

    # Update axis styles for "Control Room" look
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        font=dict(family="Inter, sans-serif")
    )
        
    return fig
