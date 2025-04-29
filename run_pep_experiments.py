#!/usr/bin/env python3
"""
Robust parallel PEPit sweep  –  pickle-safe edition
–––––––––––––––––––––––––––––––––––––––––––––––––––
• Caps BLAS threads        • spawn context
• Progress logging          • Inline exception reporting
"""

# ------------------------------------------------------------------
# 0.  Environment guards (must precede NumPy / CVXPY)
# ------------------------------------------------------------------
import os, multiprocessing as mp
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]       = "1"
os.environ["OPENBLAS_NUM_THREADS"]  = "1"

# ------------------------------------------------------------------
# 1.  Imports
# ------------------------------------------------------------------
import itertools, pickle, sys, logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from PEPit import PEP
from PEPit.functions import SmoothFunction

# ------------------------------------------------------------------
# 2.  Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] pid%(process)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 3.  Global parameters
# ------------------------------------------------------------------
L, gamma, beta, D = 1.0, 1.0, 1.0, 1.0
ALPHAS = (0.01, 0.1, 0.5, 1.0)
N_STEPS = np.arange(1, 101)
OUTFILE = "pep_results_lin_grad.pkl"

MAX_WORKERS = min(32, os.cpu_count())
CTX = mp.get_context("spawn")

# ------------------------------------------------------------------
# 4.  Worker functions (top-level => picklable)
# ------------------------------------------------------------------
def dec_worker(alpha_n):
    try:
        alpha, n = alpha_n
        problem = PEP()
        f = problem.declare_function(SmoothFunction, L=L)
        x0 = problem.set_initial_point()
        x = z = x0
        _, f0 = f.oracle(x0)

        for k in range(n):
            y = (1 - beta) * z + beta * x
            gy, _ = f.oracle(y)
            z -= gamma * gy
            c_k = 1 / (k + 1) ** alpha
            x = (1 - c_k) * x + c_k * z
            gx, _ = f.oracle(x)
            problem.set_performance_metric(gx ** 2)

        problem.set_initial_condition((f0 - f.oracle(x)[1]) <= D)
        tau = problem.solve(wrapper="cvxpy", verbose=0)
        return ("dec", alpha, n, tau)

    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        raise


def inc_worker(alpha_n):
    try:
        alpha, n = alpha_n
        problem = PEP()
        f = problem.declare_function(SmoothFunction, L=L)
        x0 = problem.set_initial_point()
        x = z = x0
        _, f0 = f.oracle(x0)

        for k in range(n):
            y = (1 - beta) * z + beta * x
            gy, _ = f.oracle(y)
            z -= gamma * gy
            c_k = (k / (k + 1)) ** alpha
            x = (1 - c_k) * x + c_k * z
            gx, _ = f.oracle(x)
            problem.set_performance_metric(gx ** 2)

        problem.set_initial_condition((f0 - f.oracle(x)[1]) <= D)
        tau = problem.solve(wrapper="cvxpy", verbose=0)
        return ("inc", alpha, n, tau)

    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        raise


def dist_worker(n):
    try:
        problem = PEP()
        f = problem.declare_function(SmoothFunction, L=L)
        x0 = problem.set_initial_point()
        x = z = x0
        _, f0 = f.oracle(x0)

        for k in range(n):
            y = (1 - beta) * z + beta * x
            gy, _ = f.oracle(y)
            z -= gamma * (k + 1) * gy
            x = (1 - 1 / (k + 1)) * x + (1 / (k + 1)) * z
            gx, _ = f.oracle(x)
            problem.set_performance_metric((x - z) ** 2)

        problem.set_initial_condition((f0 - f.oracle(x)[1]) <= D)
        tau = problem.solve(wrapper="cvxpy", verbose=0)
        return ("dist", None, n, tau)

    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        raise
        
def lin_grad_worker(n):
    try:
        problem = PEP()
        f = problem.declare_function(SmoothFunction, L=L)
        x0 = problem.set_initial_point()
        x = z = x0
        _, f0 = f.oracle(x0)

        for k in range(n):
            y = (1 - beta) * z + beta * x
            gy, _ = f.oracle(y)
            z -= gamma * (k + 1) * gy
            x = (1 - 1/(k+1)) * x + (1/(k+1)) * z
            gx, _ = f.oracle(x)
            problem.set_performance_metric(gx ** 2)

        problem.set_initial_condition((f0 - f.oracle(x)[1]) <= D)
        tau = problem.solve(wrapper="cvxpy", verbose=0)
        return ("lin_grad", None, n, tau)
    except Exception:
        import traceback
        traceback.print_exc()
        raise

# ------------------------------------------------------------------
# 5.  Driver
# ------------------------------------------------------------------
def main():
    if os.path.exists(OUTFILE):
        log.error("%s already exists - aborting.", OUTFILE)
        return

    results = {
        #"decreasing": {a: np.empty_like(N_STEPS, dtype=float) for a in ALPHAS},
        #"increasing": {a: np.empty_like(N_STEPS, dtype=float) for a in ALPHAS},
        #"distance": np.empty_like(N_STEPS, dtype=float),
        "linear_grad": np.empty_like(N_STEPS, dtype=float)
    }

    tasks_dec = list(itertools.product(ALPHAS, N_STEPS))
    tasks_inc = list(itertools.product(ALPHAS, N_STEPS))
    tasks_dist = list(N_STEPS)
    tasks_lingrad = list(N_STEPS)
    TOTAL = len(tasks_lingrad) # + len(tasks_dec) + len(tasks_inc) + len(tasks_dist) 
    done = 0

    log.info("Launching %d tasks with %d worker processes...", TOTAL, MAX_WORKERS)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS,
                             mp_context=CTX) as pool:

        futures = (
            #[pool.submit(dec_worker,  t) for t in tasks_dec]  +
            #[pool.submit(inc_worker,  t) for t in tasks_inc]  +
            #[pool.submit(dist_worker, n) for n in tasks_dist] +
            [pool.submit(lin_grad_worker, n) for n in tasks_lingrad]
        )

        for fut in as_completed(futures):
            kind, alpha, n, tau = fut.result()
            idx = n - 1
            if kind == "dec":
                results["decreasing"][alpha][idx] = tau
            elif kind == "inc":
                results["increasing"][alpha][idx] = tau
            elif kind == "dist":
                results["distance"][idx] = tau
            else:
                results["lin_grad"][idx] = tau

            done += 1
            log.info("%4d/%d finished (%s alpha=%s, n=%d)", done, TOTAL, kind, alpha, n)

    with open(OUTFILE, "wb") as fh:
        pickle.dump({"n_steps": N_STEPS, **results}, fh)
    log.info("All %d tasks done-results saved to %s", TOTAL, OUTFILE)

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
