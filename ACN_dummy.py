import numpy as np
import pandas as pd


def calculate_placement_probabilities(num_components, decay_ratio):
    """
    Calculates a truncated geometric distribution for component placement.
    Ensures that p1 > p2 > ... > pK and sum(p) = 1.
    """
    probabilities = np.array(
        [
            (1 - decay_ratio) * (decay_ratio**i) / (1 - decay_ratio**num_components)
            for i in range(num_components)
        ],
        dtype=float,
    )
    return probabilities


def allocate_devices_to_components(total_devices, probabilities):
    """
    Allocates an exact integer number of devices to each component based on the target probabilities.
    Handles fractional remainders cleanly so exactly 'total_devices' are distributed.
    """
    # Initially allocate devices based on simple rounding
    allocated_devices = np.rint(total_devices * probabilities).astype(int)

    # Calculate how many devices we are short or over due to rounding
    difference = total_devices - allocated_devices.sum()

    # Sort indices so we prioritize giving/taking from components with the highest target probability
    order = np.argsort(-probabilities)
    idx = 0

    # Distribute the remainder until we've allocated exactly 'total_devices'
    while difference != 0:
        j = order[idx % len(order)]
        if difference > 0:
            allocated_devices[j] += 1
            difference -= 1
        else:
            # Prevent taking a device away if a component only has 1 left (avoid 0 devices)
            if allocated_devices[j] > 1:
                allocated_devices[j] -= 1
                difference += 1
        idx += 1

    return allocated_devices


def calculate_theoretical_metrics(
    total_devices,
    num_components,
    decay_ratio,
    arrival_rate,
    service_rates,
    initial_delay,
    inter_component_delays,
):
    """
    Calculates queueing theory metrics using M/M/1 queue formulas for each component.
    """
    probabilities = calculate_placement_probabilities(num_components, decay_ratio)
    allocated_devices = allocate_devices_to_components(total_devices, probabilities)

    # Load is evenly split among the devices hosting the same component
    arrival_rate_per_device = arrival_rate / allocated_devices

    # Server utilization (rho) = lambda / mu
    utilization = arrival_rate_per_device / service_rates

    # Average time spent in the system (W) = 1 / (mu - lambda)
    total_time_in_system = 1.0 / (service_rates - arrival_rate_per_device)

    # Average time spent waiting in the queue (Wq) = W - (1/mu)
    time_in_queue = total_time_in_system - (1.0 / service_rates)

    # Average number of tasks in the queue (L) = rho / (1 - rho)
    avg_queue_size = utilization / (1.0 - utilization)

    # Aggregate system metrics
    total_queue_delay = time_in_queue.sum()
    total_service_delay = np.sum(1.0 / service_rates)
    total_comm_delay = initial_delay + np.sum(inter_component_delays)

    # Total average time taken by one task (T_e2e)
    end_to_end_delay = total_queue_delay + total_service_delay + total_comm_delay

    return (
        probabilities,
        allocated_devices,
        arrival_rate_per_device,
        utilization,
        total_time_in_system,
        time_in_queue,
        avg_queue_size,
        total_queue_delay,
        total_service_delay,
        total_comm_delay,
        end_to_end_delay,
    )


def run_discrete_event_simulation(
    total_devices,
    num_components,
    decay_ratio,
    arrival_rate,
    service_rates,
    initial_delay,
    inter_component_delays,
    num_tasks=50000,
    warmup_tasks=5000,
    seed=1,
):
    """
    Runs a Monte Carlo discrete-event simulation to verify theoretical predictions.
    Generates random exponential arrival and service times.
    """
    rng = np.random.default_rng(seed)

    probabilities = calculate_placement_probabilities(num_components, decay_ratio)
    allocated_devices = allocate_devices_to_components(total_devices, probabilities)
    arrival_rate_per_device = arrival_rate / allocated_devices

    mean_total_time = np.zeros(num_components)
    mean_queue_time = np.zeros(num_components)
    mean_queue_size = np.zeros(num_components)

    # Simulate each component's queue independently
    for i in range(num_components):
        server_available_time = 0.0
        current_time = 0.0
        recorded_response_times = []
        recorded_wait_times = []

        # Simulate traffic
        for j in range(num_tasks):
            # Advance time by an exponentially distributed inter-arrival gap
            current_time += rng.exponential(1.0 / arrival_rate_per_device[i])

            # Task starts processing either when it arrives, or when server is free (whichever is later)
            start_processing_time = max(current_time, server_available_time)

            # Time spent just waiting in the queue
            wait_time = start_processing_time - current_time

            # Service duration is exponentially distributed
            service_duration = rng.exponential(1.0 / service_rates[i])
            finish_time = start_processing_time + service_duration

            # Update server availability for the next incoming task
            server_available_time = finish_time

            # Only record metrics after the warmup period to ensure steady-state behavior
            if j >= warmup_tasks:
                recorded_response_times.append(
                    finish_time - current_time
                )  # Total time (queue + service)
                recorded_wait_times.append(wait_time)  # Just queue time

        # Calculate simulation averages for this component
        mean_total_time[i] = np.mean(recorded_response_times)
        mean_queue_time[i] = np.mean(recorded_wait_times)

        # Estimate average queue size using Little's Law (L = lambda * W)
        mean_queue_size[i] = arrival_rate_per_device[i] * mean_total_time[i]

    total_queue_delay = mean_queue_time.sum()
    total_service_delay = np.sum(1.0 / service_rates)
    total_comm_delay = initial_delay + np.sum(inter_component_delays)
    end_to_end_delay = total_queue_delay + total_service_delay + total_comm_delay

    return (
        probabilities,
        allocated_devices,
        arrival_rate_per_device,
        mean_total_time,
        mean_queue_time,
        mean_queue_size,
        total_queue_delay,
        total_service_delay,
        total_comm_delay,
        end_to_end_delay,
    )


if __name__ == "__main__":
    # Sample parameters (keep these fixed in the report for reproducibility)
    TOTAL_DEVICES = 100  # N: Number of devices in the network
    NUM_COMPONENTS = 5  # K: Number of components per task
    DECAY_RATIO = 0.75  # r: Controls the non-uniformity of placement
    ARRIVAL_RATE = 8.0  # lambda: Task generation rate
    SERVICE_RATES = np.array(
        [1.2, 1.2, 1.2, 1.2, 1.2]
    )  # mu: Processing capacity per component
    INITIAL_DELAY = 1.0  # d0: Comm delay to the first component
    INTER_COMPONENT_DELAYS = np.array(
        [1.0, 1.0, 1.0, 1.0]
    )  # d: Comm delays between sequential components

    # Calculate Theory
    p, n, lam_i, rho, W_th, Wq_th, L_th, Tq_th, Ts_th, Tc_th, Te2e_th = (
        calculate_theoretical_metrics(
            TOTAL_DEVICES,
            NUM_COMPONENTS,
            DECAY_RATIO,
            ARRIVAL_RATE,
            SERVICE_RATES,
            INITIAL_DELAY,
            INTER_COMPONENT_DELAYS,
        )
    )

    # Run Simulation
    _, _, _, W_sim, Wq_sim, L_sim, Tq_sim, Ts_sim, Tc_sim, Te2e_sim = (
        run_discrete_event_simulation(
            TOTAL_DEVICES,
            NUM_COMPONENTS,
            DECAY_RATIO,
            ARRIVAL_RATE,
            SERVICE_RATES,
            INITIAL_DELAY,
            INTER_COMPONENT_DELAYS,
            num_tasks=50000,
            warmup_tasks=5000,
            seed=1,
        )
    )

    # Consolidate results for presentation
    df = pd.DataFrame(
        {
            "Component": [f"C{i+1}" for i in range(NUM_COMPONENTS)],
            "Hosts n_i": n,
            "Theory W_i": W_th,
            "Sim W_i": W_sim,
            "Theory Wq_i": Wq_th,
            "Sim Wq_i": Wq_sim,
            "Theory L_i": L_th,
            "Sim L_i": L_sim,
        }
    )

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df)
    print("\nEnd-to-end delay (T_e2e):")
    print(f"Theory  T_e2e = {Te2e_th:.4f}")
    print(f"Sim     T_e2e = {Te2e_sim:.4f}")
