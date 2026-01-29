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
        
        # Re-Compute Health Status to match dashboard expected values (WARNING/CRITICAL)
        conditions = [
            (df['vibration'] > 0.8),
            (df['vibration'] > 0.6)
        ]
        choices = ['CRITICAL', 'WARNING']
        
        # Use existing if compatible, or overwrite. We overwrite to ensure consistency.
        df['health_status'] = np.select(conditions, choices, default='OK')
        
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

def create_map_figure(fleet_status, route_geojson=None, mapbox_token=None):
    """
    Creates a Mapbox scatter plot of truck positions.
    Falls back to OpenStreetMap if no token is provided.
    """
    if fleet_status.empty:
        return go.Figure()

    # Determine Map Style and Token
    if mapbox_token:
        px.set_mapbox_access_token(mapbox_token)
        map_style = "mapbox://styles/mapbox/satellite-streets-v12" # Professional look
    else:
        map_style = "open-street-map"
        
    # Create Scatter Mapbox
    fig = px.scatter_mapbox(
        fleet_status,
        lat="latitude",
        lon="longitude",
        color="operational_state",
        size="speed",
        size_max=15,
        zoom=13,
        hover_data=['truck_id', 'speed', 'engine_temperature', 'health_status'],
        title="Live Fleet GPS Tracking"
    )
    
    # Add Route Layer if available
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
                        line={'width': 2, 'color': 'blue'},
                        name='Haul Route'
                    ))
        except Exception as e:
            print(f"Error loading route for map: {e}")

    # Layout configuration
    fig.update_layout(
         mapbox_style=map_style,
         mapbox=dict(
            center=dict(
                lat=fleet_status['latitude'].mean(), 
                lon=fleet_status['longitude'].mean()
            ),
             zoom=13
         ),
         margin={"r":0,"t":40,"l":0,"b":0},
         height=600
    )
    
    if mapbox_token:
        fig.update_layout(mapbox_accesstoken=mapbox_token)

    return fig

def create_sensor_plot(df, truck_id, sensor_col, title):
    """
    Creates a time-series plot for a specific truck and sensor.
    """
    truck_data = df[df['truck_id'] == truck_id]
    
    fig = px.line(
        truck_data, 
        x='timestamp', 
        y=sensor_col, 
        title=f"Truck {truck_id}: {title}",
        color='operational_state'
    )
    
    # Highlight failure zone if Truck 7
    if truck_id == 7 and 'vibration' in sensor_col:
        fig.add_hrect(y0=0.8, y1=1.5, line_width=0, fillcolor="red", opacity=0.2, annotation_text="Critical Level")
        
    return fig
