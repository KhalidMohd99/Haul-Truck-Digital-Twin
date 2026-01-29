import simpy
import random
from simulation.config import *
from simulation.telematics import TelematicsSystem

class HaulTruck:
    def __init__(self, env, truck_id, data_log):
        self.env = env
        self.truck_id = truck_id
        self.data_log = data_log
        self.state = "QUEUE"
        self.telematics = TelematicsSystem(truck_id)
        
        # Start processes
        self.action_process = env.process(self.run())
        self.sensor_process = env.process(self.telemetry_loop())

    def run(self):
        """
        Main operational cycle: Queue -> Loading -> Haul -> Dump -> Return
        """
        # Stagger start times to avoid initial fleet synchronization
        yield self.env.timeout(random.uniform(0, 60))

        while True:
            # 1. QUEUE
            self.state = "QUEUE"
            duration = random.uniform(*QUEUE_TIME_RANGE)
            yield self.env.timeout(duration)

            # 2. LOADING
            self.state = "LOADING"
            duration = random.uniform(*LOAD_TIME_RANGE)
            yield self.env.timeout(duration)

            # 3. HAUL (Loaded Uphill)
            self.state = "HAUL"
            duration = random.uniform(*HAUL_TIME_RANGE)
            yield self.env.timeout(duration)

            # 4. DUMPING
            self.state = "DUMPING"
            duration = random.uniform(*DUMP_TIME_RANGE)
            yield self.env.timeout(duration)

            # 5. RETURN (Empty Downhill)
            self.state = "RETURN"
            duration = random.uniform(*RETURN_TIME_RANGE)
            yield self.env.timeout(duration)

    def telemetry_loop(self):
        """
        Wakes up every SAMPLE_RATE_SECONDS to record sensor data.
        """
        sampling_interval_min = SAMPLE_RATE_SECONDS / 60.0 # Sim time is in minutes
        
        while True:
            # Generate reading
            reading = self.telematics.generate_reading(
                self.state, 
                self.env.now,   # current time in minutes
                SAMPLE_RATE_SECONDS
            )
            
            # Add timestamp (Simulation Start + env.now)
            reading['timestamp'] = self.env.now
            
            self.data_log.append(reading)
            
            yield self.env.timeout(sampling_interval_min)
