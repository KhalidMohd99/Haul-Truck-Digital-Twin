import simpy, random

NUM_TRUCKS = 20

def haul_truck(env, truck_id, log):
    cycle = 0
    while True:
        log.append((env.now, truck_id, "queue"))
        yield env.timeout(random.expovariate(1/8))

        log.append((env.now, truck_id, "loading"))
        yield env.timeout(random.uniform(5,12))

        log.append((env.now, truck_id, "hauling"))
        yield env.timeout(random.uniform(25,40))

        log.append((env.now, truck_id, "dumping"))
        yield env.timeout(random.uniform(4,8))

        log.append((env.now, truck_id, "returning"))
        yield env.timeout(random.uniform(18,30))

        cycle += 1

def run_sim():
    env = simpy.Environment()
    log = []
    for i in range(NUM_TRUCKS):
        env.process(haul_truck(env, i, log))
    env.run(until=7*24*60)  # 7 days (minutes)
    return log
