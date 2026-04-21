// evaluation_simulation.cpp
// Compile with: g++ -std=c++17 -O2 evaluation_simulation.cpp -o evaluation_simulation

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::fixed;
using std::setprecision;
using std::size_t;
using std::string;
using std::vector;

struct ComponentResult {
    int component_id{};
    int host_count{};
    double placement_share{};

    double arrival_rate_per_host{};
    double utilization{};

    // Theoretical values
    double theory_W{};   // Sojourn time
    double theory_Wq{};  // Queueing delay
    double theory_L{};   // Avg. number in system
    double theory_Lq{};  // Avg. queue length

    // Simulated values
    double sim_W{};
    double sim_Wq{};
    double sim_L{};
    double sim_Lq{};

    double error_W{};
    double error_Wq{};
    double error_L{};
    double error_Lq{};
};

struct EndToEndResult {
    double theory_queue_delay{};
    double theory_service_time{};
    double theory_comm_delay{};
    double theory_e2e{};

    double sim_queue_delay{};
    double sim_service_time{};
    double sim_comm_delay{};
    double sim_e2e{};

    double error_e2e{};
};

static double percentage_error(double theoretical_value, double simulated_value) {
    if (std::abs(theoretical_value) < 1e-12) return 0.0;
    return std::abs(simulated_value - theoretical_value) / std::abs(theoretical_value) * 100.0;
}

// ------------------------------------------------------------
// 1) Normalized geometric distribution for component shares
//    p_i = (1-r) r^(i-1) / (1-r^K)
// ------------------------------------------------------------
vector<double> geometric_component_shares(int K, double r) {
    if (K <= 0) {
        throw std::invalid_argument("K must be positive.");
    }
    if (!(r > 0.0 && r < 1.0)) {
        throw std::invalid_argument("r must satisfy 0 < r < 1.");
    }

    vector<double> p(K);
    double denom = 1.0 - std::pow(r, K);

    for (int i = 0; i < K; ++i) {
        p[i] = (1.0 - r) * std::pow(r, i) / denom;
    }
    return p;
}

// ------------------------------------------------------------
// 2) Convert fractional shares into integer host counts
//    Ensures sum(n_i) = N and each n_i >= 1
// ------------------------------------------------------------
vector<int> allocate_hosts(int N, const vector<double>& p) {
    int K = static_cast<int>(p.size());
    if (N < K) {
        throw std::invalid_argument("N must be at least K so that each component can have at least one host.");
    }

    vector<int> n(K);

    // Initial rounding
    for (int i = 0; i < K; ++i) {
        n[i] = static_cast<int>(std::lround(N * p[i]));
        if (n[i] < 1) n[i] = 1;
    }

    // Fix total count to exactly N
    int diff = N - std::accumulate(n.begin(), n.end(), 0);

    // Sort indices by descending placement share
    vector<int> order(K);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return p[a] > p[b]; });

    int idx = 0;
    while (diff != 0) {
        int j = order[idx % K];

        if (diff > 0) {
            n[j] += 1;
            diff -= 1;
        } else {
            if (n[j] > 1) {
                n[j] -= 1;
                diff += 1;
            }
        }
        ++idx;
    }

    return n;
}

// ------------------------------------------------------------
// 3) Theoretical queueing metrics for each component
// ------------------------------------------------------------
vector<ComponentResult> compute_theoretical_metrics(
    int N,
    int K,
    double r,
    double global_arrival_rate,
    const vector<double>& service_rates)
{
    vector<double> p = geometric_component_shares(K, r);
    vector<int> n = allocate_hosts(N, p);

    vector<ComponentResult> results(K);

    for (int i = 0; i < K; ++i) {
        double lambda_per_host = global_arrival_rate / static_cast<double>(n[i]);
        double mu = service_rates[i];

        if (lambda_per_host >= mu) {
            throw std::runtime_error("System is unstable for component " + std::to_string(i + 1));
        }

        double rho = lambda_per_host / mu;
        double W = 1.0 / (mu - lambda_per_host);
        double Wq = W - 1.0 / mu;
        double L = rho / (1.0 - rho);
        double Lq = L - rho;

        results[i].component_id = i + 1;
        results[i].host_count = n[i];
        results[i].placement_share = p[i];
        results[i].arrival_rate_per_host = lambda_per_host;
        results[i].utilization = rho;

        results[i].theory_W = W;
        results[i].theory_Wq = Wq;
        results[i].theory_L = L;
        results[i].theory_Lq = Lq;
    }

    return results;
}

// ------------------------------------------------------------
// 4) Simulate one independent M/M/1 queue
//    - Generate Poisson arrivals
//    - Generate exponential service times
//    - Compute W, Wq, L, Lq
// ------------------------------------------------------------
struct MM1SimulationResult {
    double sim_W{};
    double sim_Wq{};
    double sim_L{};
    double sim_Lq{};
};

MM1SimulationResult simulate_mm1_queue(
    double arrival_rate,
    double service_rate,
    int total_jobs = 50000,
    int warmup_jobs = 5000,
    unsigned seed = 1)
{
    if (arrival_rate <= 0.0 || service_rate <= 0.0) {
        throw std::invalid_argument("Arrival and service rates must be positive.");
    }
    if (arrival_rate >= service_rate) {
        throw std::runtime_error("Queue is unstable because arrival_rate must be < service_rate.");
    }
    if (warmup_jobs >= total_jobs) {
        throw std::invalid_argument("warmup_jobs must be smaller than total_jobs.");
    }

    std::mt19937 rng(seed);
    std::exponential_distribution<double> inter_arrival_dist(arrival_rate);
    std::exponential_distribution<double> service_time_dist(service_rate);

    vector<double> arrival_times(total_jobs);
    vector<double> service_times(total_jobs);
    vector<double> waiting_times(total_jobs);
    vector<double> system_times(total_jobs);

    // Generate arrival times and service times
    arrival_times[0] = inter_arrival_dist(rng);
    for (int i = 1; i < total_jobs; ++i) {
        arrival_times[i] = arrival_times[i - 1] + inter_arrival_dist(rng);
    }
    for (int i = 0; i < total_jobs; ++i) {
        service_times[i] = service_time_dist(rng);
    }

    // Single-server FCFS queue
    double server_available_time = 0.0;
    for (int i = 0; i < total_jobs; ++i) {
        double start_service_time = std::max(arrival_times[i], server_available_time);
        double finish_time = start_service_time + service_times[i];

        waiting_times[i] = start_service_time - arrival_times[i];
        system_times[i] = finish_time - arrival_times[i];

        server_available_time = finish_time;
    }

    // Remove warm-up to reduce transient bias
    double sum_Wq = 0.0;
    double sum_W = 0.0;
    int count = total_jobs - warmup_jobs;

    for (int i = warmup_jobs; i < total_jobs; ++i) {
        sum_Wq += waiting_times[i];
        sum_W += system_times[i];
    }

    double sim_Wq = sum_Wq / count;
    double sim_W = sum_W / count;

    // Little's law estimates
    double sim_L = arrival_rate * sim_W;
    double sim_Lq = arrival_rate * sim_Wq;

    return {sim_W, sim_Wq, sim_L, sim_Lq};
}

// ------------------------------------------------------------
// 5) Full evaluation: theory vs simulation
// ------------------------------------------------------------
int main() {
    try {
        // -----------------------------
        // Model parameters
        // -----------------------------
        const int N = 100;
        const int K = 5;
        const double r = 0.75;
        const double global_arrival_rate = 8.0;

        // Equal service rates for all components
        vector<double> service_rates(K, 1.2);

        // Communication delays used in the analytical evaluation
        const double d0 = 1.0;
        vector<double> d_inter(K - 1, 1.0);

        // -----------------------------
        // Theoretical results
        // -----------------------------
        vector<ComponentResult> comp = compute_theoretical_metrics(
            N, K, r, global_arrival_rate, service_rates
        );

        // -----------------------------
        // Simulation results
        // -----------------------------
        for (int i = 0; i < K; ++i) {
            MM1SimulationResult sim = simulate_mm1_queue(
                comp[i].arrival_rate_per_host,
                service_rates[i],
                50000,
                5000,
                static_cast<unsigned>(i + 1)
            );

            comp[i].sim_W = sim.sim_W;
            comp[i].sim_Wq = sim.sim_Wq;
            comp[i].sim_L = sim.sim_L;
            comp[i].sim_Lq = sim.sim_Lq;

            comp[i].error_W = percentage_error(comp[i].theory_W, comp[i].sim_W);
            comp[i].error_Wq = percentage_error(comp[i].theory_Wq, comp[i].sim_Wq);
            comp[i].error_L = percentage_error(comp[i].theory_L, comp[i].sim_L);
            comp[i].error_Lq = percentage_error(comp[i].theory_Lq, comp[i].sim_Lq);
        }

        // -----------------------------
        // End-to-end delay
        // -----------------------------
        EndToEndResult e2e;

        for (const auto& c : comp) {
            e2e.theory_queue_delay += c.theory_Wq;
            e2e.sim_queue_delay += c.sim_Wq;
            e2e.theory_service_time += 1.0 / service_rates[c.component_id - 1];
            e2e.sim_service_time += 1.0 / service_rates[c.component_id - 1];
        }

        e2e.theory_comm_delay = d0 + std::accumulate(d_inter.begin(), d_inter.end(), 0.0);
        e2e.sim_comm_delay = e2e.theory_comm_delay;

        e2e.theory_e2e = e2e.theory_queue_delay + e2e.theory_service_time + e2e.theory_comm_delay;
        e2e.sim_e2e = e2e.sim_queue_delay + e2e.sim_service_time + e2e.sim_comm_delay;
        e2e.error_e2e = percentage_error(e2e.theory_e2e, e2e.sim_e2e);

        // -----------------------------
        // Print results
        // -----------------------------
        cout << fixed << setprecision(4);

        cout << "\n=== Component-wise Comparison ===\n\n";
        cout << "Comp"
             << "\tHosts"
             << "\tW_th"
             << "\tW_sim"
             << "\tWq_th"
             << "\tWq_sim"
             << "\tL_th"
             << "\tL_sim"
             << "\tErr_W(%)"
             << "\tErr_Wq(%)"
             << "\tErr_L(%)"
             << "\tErr_Lq(%)\n";

        for (const auto& c : comp) {
            cout << "C" << c.component_id
                 << "\t" << c.host_count
                 << "\t" << c.theory_W
                 << "\t" << c.sim_W
                 << "\t" << c.theory_Wq
                 << "\t" << c.sim_Wq
                 << "\t" << c.theory_L
                 << "\t" << c.sim_L
                 << "\t" << c.error_W
                 << "\t" << c.error_Wq
                 << "\t" << c.error_L
                 << "\t" << c.error_Lq
                 << "\n";
        }

        cout << "\n=== End-to-End Delay ===\n";
        cout << "Theory T_queue   = " << e2e.theory_queue_delay << "\n";
        cout << "Sim    T_queue   = " << e2e.sim_queue_delay << "\n";
        cout << "Theory T_service = " << e2e.theory_service_time << "\n";
        cout << "Sim    T_service = " << e2e.sim_service_time << "\n";
        cout << "Theory T_comm    = " << e2e.theory_comm_delay << "\n";
        cout << "Sim    T_comm    = " << e2e.sim_comm_delay << "\n";
        cout << "Theory T_e2e     = " << e2e.theory_e2e << "\n";
        cout << "Sim    T_e2e     = " << e2e.sim_e2e << "\n";
        cout << "Error T_e2e(%)   = " << e2e.error_e2e << "\n";

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}