#!/usr/bin/env python3
"""
Run all PEPit experiments and dump the results to pep_results.pkl
_______________________________________________________________
Usage (inside tmux or the shell):   python run_pep_experiments.py
"""

import itertools, pickle, os, numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from PEPit import PEP
from PEPit.functions import SmoothFunction


# ------------------------------------------------------------------
# --- parameters you might want to tweak ---------------------------
L       = 1.0          # smoothness
gamma   = 1.0          # L·η ≤ 1
beta    = 1.0          # interpolation parameter
D       = 1.0          # initial distance bound
ALPHAS  = (0.01, 0.1, 0.5, 1.0)
N_STEPS = np.arange(1, 101)
OUTFILE = "pep_results.pkl"
# ------------------------------------------------------------------


def decreasing_c_worker(alpha_n):
    """One run for c_{t+1} = (1/(t+1))^α."""
    alpha, n = alpha_n
    problem  = PEP()
    f        = problem.declare_function(SmoothFunction, L=L)
    x0       = problem.set_initial_point()
    x = z    = x0
    _, f0    = f.oracle(x0)

    for k in range(n):
        y, _  = (1 - beta) * z + beta * x, None
        gy, _ = f.oracle(y)
        z     = z - gamma * gy
        c_k   = 1 / (k + 1) ** alpha
        x     = (1 - c_k) * x + c_k * z
        gx, _ = f.oracle(x)
        problem.set_performance_metric(gx ** 2)

    problem.set_initial_condition((f0 - f.oracle(x)[1]) <= D)
    tau = problem.solve(wrapper="cvxpy", verbose=0)
    return ("dec", alpha, n, tau)


def increasing_c_worker(alpha_n):
    """One run for c_{t+1} = (t/(t+1))^α."""
    alpha, n = alpha_n
    problem  = PEP()
    f        = problem.declare_function(SmoothFunction, L=L)
    x0       = problem.set_initial_point()
    x = z    = x0
    _, f0    = f.oracle(x0)

    for k in range(n):
        y, _  = (1 - beta) * z + beta * x, None
        gy, _ = f.oracle(y)
        z     = z - gamma * gy
        c_k   = (k / (k + 1)) ** alpha
        x     = (1 - c_k) * x + c_k * z
        gx, _ = f.oracle(x)
        problem.set_performance_metric(gx ** 2)

    problem.set_initial_condition((f0 - f.oracle(x)[1]) <= D)
    tau = problem.solve(wrapper="cvxpy", verbose=0)
    return ("inc", alpha, n, tau)


def distance_worker(n):
    """Worst-case ‖zₙ − xₙ‖² when η_t = (t+1)/L."""
    problem  = PEP()
    f        = problem.declare_function(SmoothFunction, L=L)
    x0       = problem.set_initial_point()
    x = z    = x0
    _, f0    = f.oracle(x0)

    for k in range(n):
        y, _  = (1 - beta) * z + beta * x, None
        gy, _ = f.oracle(y)
        z     = z - gamma * (k + 1) * gy          # linear step
        x     = (1 - 1 / (k + 1)) * x + (1 / (k + 1)) * z
        gx, _ = f.oracle(x)
        problem.set_performance_metric((x - z) ** 2)

    problem.set_initial_condition((f0 - f.oracle(x)[1]) <= D)
    tau = problem.solve(wrapper="cvxpy", verbose=0)
    return ("dist", None, n, tau)


def main():
    # skip everything if results already exist
    if os.path.exists(OUTFILE):
        print(f"{OUTFILE} already present – nothing to do.")
        return

    tasks_dec = list(itertools.product(ALPHAS, N_STEPS))
    tasks_inc = list(itertools.product(ALPHAS, N_STEPS))
    tasks_dist = N_STEPS

    results = {
        "decreasing": {a: np.empty_like(N_STEPS, dtype=float) for a in ALPHAS},
        "increasing": {a: np.empty_like(N_STEPS, dtype=float) for a in ALPHAS},
        "distance":   np.empty_like(N_STEPS, dtype=float),
    }

    with ProcessPoolExecutor() as pool:
        futures = (
            [pool.submit(decreasing_c_worker, t) for t in tasks_dec] +
            [pool.submit(increasing_c_worker, t) for t in tasks_inc] +
            [pool.submit(distance_worker,       n) for n in tasks_dist]
        )

        for fut in as_completed(futures):
            kind, alpha, n, tau = fut.result()
            idx = n - 1  # zero-based
            if kind == "dec":
                results["decreasing"][alpha][idx] = tau
            elif kind == "inc":
                results["increasing"][alpha][idx] = tau
            else:
                results["distance"][idx] = tau

    with open(OUTFILE, "wb") as fh:
        pickle.dump({"n_steps": N_STEPS, **results}, fh)
    print(f"Saved results to {OUTFILE}")


if __name__ == "__main__":
    main()
