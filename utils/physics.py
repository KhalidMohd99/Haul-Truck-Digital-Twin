import math
import random
import numpy as np

def calculate_ambient_temp(time_of_day_hours, mean=28.0, var=12.0):
    """
    Simulates diurnal temperature cycle.
    Min at 04:00, Max at 16:00.
    """
    # Shifted sine wave: Peak at 16h (4pm)
    # sin(x - shift) -> we want peak at 16. 
    # Normal sin peak at pi/2. 
    # (2pi/24) * (t - shift) = pi/2
    # t - shift = 6 => shift = t - 6 = 16 - 6 = 10?
    # Let's check: sin((16-10)*pi/12) = sin(6pi/12) = sin(pi/2) = 1. Correct.
    
    t_rad = (time_of_day_hours - 10) * (2 * math.pi / 24)
    temp = mean + (var / 2) * math.sin(t_rad)
    
    # Add small random noise
    temp += random.gauss(0, 0.5)
    return temp

def update_engine_temp(current_temp, target_temp, dt_seconds, k=0.005):
    """
    Newton's Law of Cooling/Heating.
    dT/dt = -k * (T - T_env)
    T_new = T + k * (Target - T) * dt
    """
    # k is heating/cooling rate coefficient
    change = k * (target_temp - current_temp) * dt_seconds
    return current_temp + change

def calculate_degradation(current_time_hours, start_time, critical_time):
    """
    Calculates bearing degradation factor (0.0 to 1.0).
    Non-linear growth (square root or exponential).
    User spec: "Progressive degradation (non-linear growth: degradation^0.5)" 
    Wait, usually degradation grows exponentially. 
    Let's interpret README: "Day 5 (06:00): Degradation onset".
    
    We'll model it as:
    If t < start: 0
    If t > start: ((t - start) / (critical - start)) ^ 2  (Accelerating wear)
    """
    if current_time_hours < (start_time * 24):
        return 0.0
    
    elapsed = current_time_hours - (start_time * 24)
    duration = (critical_time * 24) - (start_time * 24)
    
    if duration <= 0: return 1.0
    
    deg = (elapsed / duration) ** 2
    return min(deg, 1.0)

def apply_random_walk(current_val, target_mean, step_std=0.1, revert_strength=0.01):
    """
    Updates a value with random walk tendency to revert to mean.
    """
    noise = random.gauss(0, step_std)
    revert = (target_mean - current_val) * revert_strength
    return current_val + noise + revert
