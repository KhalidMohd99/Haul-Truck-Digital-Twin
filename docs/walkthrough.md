# Haul Truck Digital Twin - Walkthrough

I have successfully implemented the physics-based fleet simulation, resolved installation issues, and deployed an interactive Streamlit dashboard now enhanced with **GPS Tracking**.

## 🏗️ Architecture Implemented

*   **`simulation/`**: Core logic including `main.py` (runner), `truck.py` (SimPy agent), `telematics.py` (physics engine).
*   **`web_dashboard/`**: Interactive visualization tool.
    *   `app.py`: Main Streamlit application.
    *   `utils.py`: Data loaders and Plotly charts.
*   **`docs/`**: Project documentation (`implementation_plan.md`, `task.md`).

## 📊 Results Verification

I executed `python -m simulation.main` and verified the output:

*   **Dataset**: `data/mansourah_haul_truck_telemetry.csv` (~36 MB)
*   **Records**: 403,200 rows.
*   **Physics**: Truck 7 vibration drift and engine thermal cycles are present.

## 📈 Dashboard Features

The new `web_dashboard` provides:
1.  **Fleet Map**: Visualizes truck positions along the North-South corridor using **Real GPS Data**.
    *   **Mapbox Support**: Optional Satellite view (enter token in sidebar).
    *   **Route Overlay**: Shows the haul road path.
2.  **Live Telemetry**: Time-series plots for Engine Temp, Vibration, and Tire Pressure.
3.  **Health Monitor**: Highlights Truck 7's "Warning" and "Critical" states as vibration increases.

## 🚀 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Generate Data** (Optional if GPS data provided):
    ```bash
    python -m simulation.main
    ```
3.  **Launch Dashboard**:
    Use the following command:
    ```bash
    python -m streamlit run web_dashboard/app.py
    ```
