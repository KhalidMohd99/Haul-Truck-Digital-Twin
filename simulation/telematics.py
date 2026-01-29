import random
from simulation.config import *
from utils.physics import (
    calculate_ambient_temp, 
    update_engine_temp, 
    calculate_degradation,
    apply_random_walk
)

class TelematicsSystem:
    def __init__(self, truck_id, initial_time_hours=0):
        self.truck_id = truck_id
        
        # Initial State
        self.engine_temp = BASE_ENGINE_TEMP_C
        self.tire_pressure = 98.0  # psi
        self.fuel_level = 100.0    # %
        self.vibration = VIBRATION_NORMAL_BASE
        
        # Tracking
        self.cumulative_fuel_consumed = 0.0
        
    def generate_reading(self, state, time_minutes, dt_seconds):
        """
        Generates a dictionary of sensor readings for the current time step.
        """
        time_hours = time_minutes / 60.0
        
        # 1. Ambient Conditions
        ambient_temp = calculate_ambient_temp(time_hours % 24, AMBIENT_TEMP_MEAN_C, AMBIENT_TEMP_VAR_C)
        
        # 2. Engine Temperature Logic
        target_temp = BASE_ENGINE_TEMP_C
        if state == "HAUL":
            target_temp = MAX_ENGINE_TEMP_C  # High load
        elif state == "RETURN":
            target_temp = BASE_ENGINE_TEMP_C + 10 # Moderate
        else:
            target_temp = BASE_ENGINE_TEMP_C # Idle
            
        # Add ambient influence
        target_temp += (ambient_temp - 25) * 0.2
        
        # Update temp physics
        heating_rate = 0.02 if state == "HAUL" else 0.01
        self.engine_temp = update_engine_temp(self.engine_temp, target_temp, dt_seconds, k=heating_rate)
        
        # 3. Vibration Logic
        base_vib = VIBRATION_NORMAL_BASE
        if state == "HAUL": base_vib += VIBRATION_LOADED_OFFSET
        elif state == "LOADING": base_vib += 0.1
        elif state == "DUMPING": base_vib += 0.15
        
        # Truck 7 Failure Injection
        degradation = 0.0
        if self.truck_id == FAILING_TRUCK_ID:
            degradation = calculate_degradation(time_hours, FAILURE_START_DAY, FAILURE_CRITICAL_DAY)
            # Add gradually increasing vibration noise and offset
            failure_impact = degradation * VIBRATION_FAILING_FACTOR
            base_vib += failure_impact * random.uniform(0.8, 1.2)
            
            # Start fluctuating wilder near end
            if degradation > 0.8:
                base_vib += random.gauss(0, 0.2)

        self.vibration = max(0, base_vib + random.gauss(0, 0.02))
        
        # 4. Tire Pressure (Fluctuates with temp and random walk)
        # Pressure increases with temp (PV=nRT approx)
        temp_factor = (self.engine_temp - 85) * 0.1
        target_pressure = 98.0 + temp_factor
        self.tire_pressure = apply_random_walk(self.tire_pressure, target_pressure, step_std=0.05)
        
        # 5. Speed (Simplified based on state for telemetry)
        speed = 0.0
        if state == "HAUL": speed = random.normalvariate(15, 2)  # Slow uphill
        elif state == "RETURN": speed = random.normalvariate(45, 5) # Fast downhill
        elif state == "QUEUE": speed = 0
        elif state == "LOADING": speed = 0
        elif state == "DUMPING": speed = 0
        
        # 6. Fuel Rate (L/h)
        fuel_rate = 2.0 # Idle
        if state == "HAUL": fuel_rate = 45.0
        elif state == "RETURN": fuel_rate = 12.0
        elif state == "DUMPING": fuel_rate = 8.0
        
        self.cumulative_fuel_consumed += (fuel_rate * (dt_seconds/3600))

        # 7. Remaining Useful Life (RUL)
        # For training targets. 
        # For Truck 7: Hours until end of simulation (assuming failure at end)
        # or specific failure point. 
        # Let's say failure point is Day 7.0 (end of sim).
        rul = 1000.0 # Default high
        if self.truck_id == FAILING_TRUCK_ID:
            failure_time_h = FAILURE_CRITICAL_DAY * 24
            remaining_h = failure_time_h - time_hours
            rul = max(0, remaining_h)

        return {
            "truck_id": self.truck_id,
            "timestamp": None, # To be filled by runner
            "operational_state": state,
            "engine_temperature": round(self.engine_temp, 2),
            "vibration": round(self.vibration, 4),
            "tire_pressure": round(self.tire_pressure, 2),
            "speed": round(speed, 2),
            "fuel_rate": round(fuel_rate, 2),
            "bearing_degradation": round(degradation, 4),
            "rul_hours": round(rul, 2)
        }
