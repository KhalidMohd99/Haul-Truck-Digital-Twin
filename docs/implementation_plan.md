# Implementation Plan - Haul Truck Digital Twin

## Goal Description
Simulate a fleet of 20 Caterpillar-class haul trucks at Mansourah–Massarah Gold Mine for 7 days. Generate labeled time-series telemetry data (CSV) with a realistic developing bearing failure in Truck 7 for LSTM anomaly detection training. **ADDITIONALLY: Create a Streamlit Dashboard to visualize the live data.**

## User Review Required
> [!IMPORTANT]
> **Environment Fix**: The user attempted to create a virtual environment but encountered issues with activation commands in PowerShell. We will ignore the broken `.venv`/`venv` and use `pip install -r requirements.txt` directly or a fresh environment with correct PowerShell activation. The immediate next step is to ensure `streamlit` is installed and the dashboard runs.

## Proposed Changes

### Dashboard Implementation
Create a new directory `web_dashboard/` containing:
- `app.py`: Main Streamlit application.
- `utils.py`: Helper functions for loading and plotting data.

#### `web_dashboard/app.py`
- **Sidebar**: Select Truck ID, Date Range.
- **Main View**:
    - **Fleet Overview**: Map of current positions (simulated lat/lon along corridor).
    - **Truck Detail**: Time-series plots for selected truck (Engine Temp, Vibration, Speed).
    - **Health Monitor**: Highlight Truck 7's degradation vs Healthy Trucks.

### GPS Data Integration [NEW]
- **Data Source**: Re-point dashboard to use `data/GPS Enhancement/mansourah_haul_truck_telemetry_with_gps.csv` which contains `latitude`, `longitude`.
- **Map Visualization**: Update `utils.py` to use `data/routes.geojson` for plotting the route.
- **Map Engine**: Use `plotly.scatter_mapbox`.
    - **Token Policy**: Make Mapbox token OPTIONAL.
    - **Fallback**: If no token is provided, use `open-street-map` style.
    - **Future Proofing**: Structure code to allow swapping for Folium later.

### Project Structure (Updated)
```text
haul-truck-digital-twin/
├── web_dashboard/      # [NEW]
│   ├── app.py
│   └── utils.py
├── simulation/
│   ... (existing)
├── ml/
│   ... (existing)
├── utils/
│   ... (existing)
├── data/
│   ├── GPS Enhancement/
│   │   └── mansourah_haul_truck_telemetry_with_gps.csv
│   ├── routes.geojson
│   └── (Target for CSV output)
├── requirements.txt    # [UPDATE] Add streamlit, plotly
└── README.md
```

### Simulation Logic (Existing - No Changes)
- `simulation/config.py`: Constants.
- `simulation/truck.py`: State machine.
- `simulation/telematics.py`: Physics logic.

## Verification Plan

### Automated Tests
- Run `streamlit run web_dashboard/app.py` and verify it loads without error.
- Check that the dashboard correctly loads `data/mansourah_haul_truck_telemetry.csv`.

### Manual Verification
- Interactively select "Truck 7" in the dashboard.
- Verify the "Vibration" plot shows the drift starting on Day 5.
- **GPS Verification**: Verify trucks appear on the map at coordinates from the CSV.
