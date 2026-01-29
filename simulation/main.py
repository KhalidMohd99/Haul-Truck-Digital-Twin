import simpy
import pandas as pd
import random
from simulation.config import *
from simulation.truck import HaulTruck
from datetime import datetime, timedelta

def run_simulation():
    print(f"Starting Simulation: {NUM_TRUCKS} Trucks for {SIMULATION_DAYS} Days...")
    
    # Initialize SimPy Environment
    env = simpy.Environment()
    data_log = []
    
    # Create Fleet
    trucks = []
    for i in range(1, NUM_TRUCKS + 1):
        trucks.append(HaulTruck(env, i, data_log))
        
    # Run Simulation
    env.run(until=SIMULATION_DURATION_MINUTES)
    
    print(f"Simulation Complete. Generated {len(data_log)} records.")
    
    # Process Data
    df = pd.DataFrame(data_log)
    
    # Convert simulation minutes to actual datetime
    # Let's start simulation at 2024-01-01 00:00:00
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    df['timestamp'] = df['timestamp'].apply(lambda m: start_time + timedelta(minutes=m))
    
    # Sort and Save
    df = df.sort_values(by=['timestamp', 'truck_id'])
    
    output_path = "data/mansourah_haul_truck_telemetry.csv"
    # Ensure data dir exists
    import os
    os.makedirs("data", exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    run_simulation()
