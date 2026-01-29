# Simulation Configuration

# Simulation Parameters
NUM_TRUCKS = 20
SIMULATION_DAYS = 7
SIMULATION_DURATION_MINUTES = SIMULATION_DAYS * 24 * 60
SAMPLE_RATE_SECONDS = 30
RANDOM_SEED = 42

# Truck Specs (Caterpillar 793F Class)
MAX_PAYLOAD_TONS = 231
EMPTY_WEIGHT_TONS = 122
MAX_SPEED_KMH = 60.0

# Operational Cycle Parameters (Minutes)
# Distributions modeled as (mu, sigma) for Gaussian or (min, max) for Uniform
QUEUE_TIME_RANGE = (1, 5)      # Waiting for shovel
LOAD_TIME_RANGE = (3, 6)       # Loading limits
HAUL_TIME_RANGE = (12, 18)     # Uphill loaded
DUMP_TIME_RANGE = (2, 4)       # Dumping
RETURN_TIME_RANGE = (10, 15)   # Downhill empty

# Physics Baselines
BASE_ENGINE_TEMP_C = 85.0
MAX_ENGINE_TEMP_C = 115.0
AMBIENT_TEMP_MEAN_C = 28.0
AMBIENT_TEMP_VAR_C = 12.0      # Day/Night swing amplitude

# Truck 7 Failure Injection
FAILING_TRUCK_ID = 7
FAILURE_START_DAY = 5.0        # Start degrading at Day 5
FAILURE_CRITICAL_DAY = 6.5     # Critical by Day 6.5
# Vibration Levels (g)
VIBRATION_NORMAL_BASE = 0.20
VIBRATION_LOADED_OFFSET = 0.30 # Added when heavy
VIBRATION_FAILING_FACTOR = 3.0 # Multiplier at max failure
