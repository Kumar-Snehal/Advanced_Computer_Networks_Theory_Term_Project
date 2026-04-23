# evaluation_simulation.py
import numpy as np
import pandas as pd


def geometric_p(K, r):
    p = np.array([(1 - r) * (r**i) / (1 - r**K) for i in range(K)], dtype=float)
    return p


def allocate_hosts(N, p):
    n = np.rint(N * p).astype(int)
    diff = N - n.sum()
    order = np.argsort(-p)  # largest p first
    idx = 0
    while diff != 0:
        j = order[idx % len(order)]
        if diff > 0:
            n[j] += 1
            diff -= 1
        else:
            if n[j] > 1:
                n[j] -= 1
                diff += 1
        idx += 1
    return n


def theory(N, K, r, lam, mu, d0, d):
    p = geometric_p(K, r)
    n = allocate_hosts(N, p)
    lam_i = lam / n
    rho = lam_i / mu
    W = 1.0 / (mu - lam_i)
    Wq = W - 1.0 / mu
    L = rho / (1.0 - rho)
    T_queue = Wq.sum()
    T_service = np.sum(1.0 / mu)
    T_comm = d0 + np.sum(d)
    T_e2e = T_queue + T_service + T_comm
    return p, n, lam_i, rho, W, Wq, L, T_queue, T_service, T_comm, T_e2e


def simulate_mm1_per_component(
    N, K, r, lam, mu, d0, d, n_jobs=50000, warmup=5000, seed=1
):
    rng = np.random.default_rng(seed)
    p = geometric_p(K, r)
    n = allocate_hosts(N, p)
    lam_i = lam / n

    mean_W = np.zeros(K)
    mean_Wq = np.zeros(K)
    mean_L = np.zeros(K)

    for i in range(K):
        avail = 0.0
        t = 0.0
        resp = []
        wait = []
        for j in range(n_jobs):
            t += rng.exponential(1.0 / lam_i[i])
            start = max(t, avail)
            w = start - t
            s = rng.exponential(1.0 / mu[i])
            finish = start + s
            avail = finish
            if j >= warmup:
                resp.append(finish - t)
                wait.append(w)
        mean_W[i] = np.mean(resp)
        mean_Wq[i] = np.mean(wait)
        mean_L[i] = lam_i[i] * mean_W[i]  # Little's law estimate

    T_queue = mean_Wq.sum()
    T_service = np.sum(1.0 / mu)
    T_comm = d0 + np.sum(d)
    T_e2e = T_queue + T_service + T_comm

    return p, n, lam_i, mean_W, mean_Wq, mean_L, T_queue, T_service, T_comm, T_e2e


if __name__ == "__main__":
    # Sample parameters (keep these fixed in the report for reproducibility)
    N = 100
    K = 5
    r = 0.75
    lam = 8.0
    mu = np.array([1.2, 1.2, 1.2, 1.2, 1.2])
    d0 = 1.0
    d = np.array([1.0, 1.0, 1.0, 1.0])

    p, n, lam_i, rho, W_th, Wq_th, L_th, Tq_th, Ts_th, Tc_th, Te2e_th = theory(
        N, K, r, lam, mu, d0, d
    )
    _, _, _, W_sim, Wq_sim, L_sim, Tq_sim, Ts_sim, Tc_sim, Te2e_sim = (
        simulate_mm1_per_component(
            N, K, r, lam, mu, d0, d, n_jobs=50000, warmup=5000, seed=1
        )
    )

    df = pd.DataFrame(
        {
            "Component": [f"C{i+1}" for i in range(K)],
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
    print("\nEnd-to-end delay:")
    print(f"Theory  T_e2e = {Te2e_th:.4f}")
    print(f"Sim     T_e2e = {Te2e_sim:.4f}")
