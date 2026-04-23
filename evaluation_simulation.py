import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_placement_probabilities(num_components, decay_ratio):
    """Calculates a truncated geometric distribution for component placement."""
    probabilities = np.array(
        [
            (1 - decay_ratio) * (decay_ratio**i) / (1 - decay_ratio**num_components)
            for i in range(num_components)
        ],
        dtype=float,
    )
    return probabilities

def allocate_devices_to_components(total_devices, probabilities):
    """Allocates an exact integer number of devices to each component."""
    allocated_devices = np.rint(total_devices * probabilities).astype(int)
    difference = total_devices - allocated_devices.sum()
    order = np.argsort(-probabilities)
    idx = 0
    while difference != 0:
        j = order[idx % len(order)]
        if difference > 0:
            allocated_devices[j] += 1
            difference -= 1
        else:
            if allocated_devices[j] > 1:
                allocated_devices[j] -= 1
                difference += 1
        idx += 1
    return allocated_devices

def calculate_theoretical_metrics(total_devices, num_components, decay_ratio, arrival_rate, service_rates, initial_delay, inter_component_delays):
    """Calculates queueing theory metrics using M/M/1 formulas."""
    probabilities = calculate_placement_probabilities(num_components, decay_ratio)
    allocated_devices = allocate_devices_to_components(total_devices, probabilities)
    arrival_rate_per_device = arrival_rate / allocated_devices
    utilization = arrival_rate_per_device / service_rates
    total_time_in_system = 1.0 / (service_rates - arrival_rate_per_device)
    time_in_queue = total_time_in_system - (1.0 / service_rates)
    avg_queue_size = utilization / (1.0 - utilization)
    
    total_queue_delay = time_in_queue.sum()
    total_service_delay = np.sum(1.0 / service_rates)
    total_comm_delay = initial_delay + np.sum(inter_component_delays)
    end_to_end_delay = total_queue_delay + total_service_delay + total_comm_delay

    return (probabilities, allocated_devices, arrival_rate_per_device, utilization, total_time_in_system, time_in_queue, avg_queue_size, total_queue_delay, total_service_delay, total_comm_delay, end_to_end_delay)

def run_discrete_event_simulation(total_devices, num_components, decay_ratio, arrival_rate, service_rates, initial_delay, inter_component_delays, num_tasks=50000, warmup_tasks=5000, seed=1):
    """Runs a Monte Carlo discrete-event simulation."""
    rng = np.random.default_rng(seed)
    probabilities = calculate_placement_probabilities(num_components, decay_ratio)
    allocated_devices = allocate_devices_to_components(total_devices, probabilities)
    arrival_rate_per_device = arrival_rate / allocated_devices

    mean_total_time = np.zeros(num_components)
    mean_queue_time = np.zeros(num_components)

    for i in range(num_components):
        server_available_time, current_time = 0.0, 0.0
        recorded_response_times, recorded_wait_times = [], []
        for j in range(num_tasks):
            current_time += rng.exponential(1.0 / arrival_rate_per_device[i])
            start_processing_time = max(current_time, server_available_time)
            wait_time = start_processing_time - current_time
            service_duration = rng.exponential(1.0 / service_rates[i])
            finish_time = start_processing_time + service_duration
            server_available_time = finish_time
            if j >= warmup_tasks:
                recorded_response_times.append(finish_time - current_time)
                recorded_wait_times.append(wait_time)
        mean_total_time[i] = np.mean(recorded_response_times)
        mean_queue_time[i] = np.mean(recorded_wait_times)

    total_queue_delay = mean_queue_time.sum()
    total_service_delay = np.sum(1.0 / service_rates)
    total_comm_delay = initial_delay + np.sum(inter_component_delays)
    end_to_end_delay = total_queue_delay + total_service_delay + total_comm_delay

    return (probabilities, allocated_devices, arrival_rate_per_device, mean_total_time, mean_queue_time, total_queue_delay, total_service_delay, total_comm_delay, end_to_end_delay)

def generate_visualizations(df, total_q, total_s, total_c, N, K, r, mu):
    """Generates result graphs for presentation."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Validation: Theory vs Simulation (Sojourn Time)
    df_melted = df.melt(id_vars='Component', value_vars=['Theory W_i', 'Sim W_i'], var_name='Type', value_name='Time')
    sns.barplot(ax=axes[0, 0], data=df_melted, x='Component', y='Time', hue='Type', palette='viridis')
    axes[0, 0].set_title(r'Validation: Theory vs. Simulation Sojourn Time ($W_i$)', fontsize=14)
    axes[0, 0].set_ylabel('Avg Time in System')

    # 2. Impact of Non-Uniformity (Bottleneck Trend)
    sns.lineplot(ax=axes[0, 1], x=range(1, K+1), y=df['Sim Wq_i'], marker='o', color='crimson', lw=2)
    axes[0, 1].set_title('Bottleneck Identification: Queueing Delay Trend', fontsize=14)
    axes[0, 1].set_xlabel('Component Index')
    axes[0, 1].set_ylabel(r'Wait Time ($W_{q,i}$)')

    # 3. Delay Breakdown (Pie Chart)
    labels = ['Queueing Delay', 'Service Time', 'Comm Delay']
    sizes = [total_q, total_s, total_c]
    axes[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    axes[1, 0].set_title('Total End-to-End Delay Breakdown', fontsize=14)

    # 4. Stability Analysis (Modeled Load Sensitivity)
    lam_range = np.linspace(1, 9, 20) 
    theoretical_e2e = []
    for l in lam_range:
        # Using a raw string for lambda in title and xlabel
        _, _, _, _, _, _, _, _, _, _, e2e = calculate_theoretical_metrics(N, K, r, l, mu, 1.0, np.array([1.0]*(K-1)))
        theoretical_e2e.append(e2e)
    
    sns.lineplot(ax=axes[1, 1], x=lam_range, y=theoretical_e2e, color='forestgreen', lw=2)
    axes[1, 1].set_title(r'System Stability: Task Arrival Rate ($\lambda$) vs. $T_{e2e}$', fontsize=14)
    axes[1, 1].set_xlabel(r'Global Arrival Rate ($\lambda$)')
    axes[1, 1].set_ylabel('End-to-End Delay')

    plt.tight_layout()
    # Save the file instead of showing it to avoid UserWarning
    plt.savefig('simulation_results.png')
    print("Graphs successfully generated and saved as 'simulation_results.png'.")

if __name__ == "__main__":
    # Parameters
    TOTAL_DEVICES = 100
    NUM_COMPONENTS = 5
    DECAY_RATIO = 0.75
    ARRIVAL_RATE = 8.0
    SERVICE_RATES = np.array([1.2] * NUM_COMPONENTS)
    INITIAL_DELAY = 1.0
    INTER_COMPONENT_DELAYS = np.array([1.0] * (NUM_COMPONENTS - 1))

    # Calculate Theory
    p, n, lam_i, rho, W_th, Wq_th, L_th, Tq_th, Ts_th, Tc_th, Te2e_th = calculate_theoretical_metrics(
        TOTAL_DEVICES, NUM_COMPONENTS, DECAY_RATIO, ARRIVAL_RATE, SERVICE_RATES, INITIAL_DELAY, INTER_COMPONENT_DELAYS)

    # Run Simulation
    _, _, _, W_sim, Wq_sim, Tq_sim, Ts_sim, Tc_sim, Te2e_sim = run_discrete_event_simulation(
        TOTAL_DEVICES, NUM_COMPONENTS, DECAY_RATIO, ARRIVAL_RATE, SERVICE_RATES, INITIAL_DELAY, INTER_COMPONENT_DELAYS)

    # Consolidate Data
    df = pd.DataFrame({
        "Component": [f"C{i+1}" for i in range(NUM_COMPONENTS)],
        "Hosts n_i": n,
        "Theory W_i": W_th,
        "Sim W_i": W_sim,
        "Theory Wq_i": Wq_th,
        "Sim Wq_i": Wq_sim
    })

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df)
    print(f"\nEnd-to-end delay (T_e2e): Theory={Te2e_th:.4f}, Sim={Te2e_sim:.4f}")

    # Generate Graphs
    generate_visualizations(df, Tq_sim, Ts_sim, Tc_sim, TOTAL_DEVICES, NUM_COMPONENTS, DECAY_RATIO, SERVICE_RATES)