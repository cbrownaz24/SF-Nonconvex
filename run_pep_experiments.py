#!/usr/bin/env python3
"""
Robust parallel PEPit sweep
––––––––––––––––––––––––––––
• Capped BLAS threads  • spawn context
• Guarded workers      • Live progress logging
"""

# ------------------------------------------------------------------
# 0.  Environment guards
# ------------------------------------------------------------------
import os, multiprocessing as mp
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]       = "1"
os.environ["OPENBLAS_NUM_THREADS"]  = "1"

# ------------------------------------------------------------------
# 1.  Imports
# ------------------------------------------------------------------
import itertools, pickle, sys, logging, datetime
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from PEPit import PEP
from PEPit.functions import SmoothFunction


# ------------------------------------------------------------------
# 2.  Logging setup
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%Y-%m-%d %H:%M:%S] pid%(process)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 3.  Parameters
# ------------------------------------------------------------------
L, gamma, beta, D = 1.0, 1.0, 1.0, 1.0
ALPHAS = (0.01, 0.1, 0.5, 1.0)
N_STEPS = np.arange(1, 101)
OUTFILE = "pep_results.pkl"

MAX_WORKERS = min(32, os.cpu_count())
CTX = mp.get_context("spawn")


# ------------------------------------------------------------------
# 4.  Guard decorator
# ------------------------------------------------------------------
def _guard(fn):
    def wrapped(arg):
        try:
            return fn(arg)
        except Exception:  # print traceback inside worker
            import traceback
            traceback.print_exc()
            sys.stdout.flush(); sys.stderr.flush()
            raise
    return wrapped


# ------------------------------------------------------------------
# 5.  Worker definitions  (identical maths, just logging flush)
# ------------------------------------------------------------------
@_guard
def dec_worker(alpha_n):
    alpha, n = alpha_n
    problem = PEP(); f = problem.declare_function(SmoothFunction, L=L)
    x0 = problem.set_initial_point(); x = z = x0; _, f0 = f.oracle(x0)
    for k in range(n):
        y = (1 - beta) * z + beta * x
        gy, _ = f.oracle(y); z -= gamma * gy
        c_k = 1 / (k + 1) ** alpha
        x = (1 - c_k) * x + c_k * z
        gx, _ = f.oracle(x)
        problem.set_performance_metric(gx ** 2)
    problem.set_initial_condition((f0 - f.oracle(x)[1]) <= D)
    tau = problem.solve(wrapper="cvxpy", verbose=0)
    return ("dec", alpha, n, tau)


@_guard
def inc_worker(alpha_n):
    alpha, n = alpha_n
    problem = PEP(); f = problem.declare_function(SmoothFunction, L=L)
    x0 = problem.set_initial_point(); x = z = x0; _, f0 = f.oracle(x0)
    for k in range(n):
        y = (1 - beta) * z + beta * x
        gy, _ = f.oracle(y); z -= gamma * gy
        c_k = (k / (k + 1)) ** alpha
        x = (1 - c_k) * x + c_k * z
        gx, _ = f.oracle(x)
        problem.set_performance_metric(gx ** 2)
    problem.set_initial_condition((f0 - f.oracle(x)[1]) <= D)
    tau = problem.solve(wrapper="cvxpy", verbose=0)
    return ("inc", alpha, n, tau)


@_guard
def dist_worker(n):
    problem = PEP(); f = problem.declare_function(SmoothFunction, L=L)
    x0 = problem.set_initial_point(); x = z = x0; _, f0 = f.oracle(x0)
    for k in range(n):
        y = (1 - beta) * z + beta * x
        gy, _ = f.oracle(y); z -= gamma * (k + 1) * gy
        x = (1 - 1 / (k + 1)) * x + (1 / (k + 1)) * z
        gx, _ = f.oracle(x)
        problem.set_performance_metric((x - z) ** 2)
    problem.set_initial_condition((f0 - f.oracle(x)[1]) <= D)
    tau = problem.solve(wrapper="cvxpy", verbose=0)
    return ("dist", None, n, tau)


# ------------------------------------------------------------------
# 6.  Driver
# ------------------------------------------------------------------
def main():
    if os.path.exists(OUTFILE):
        log.error("%s already exists – aborting.", OUTFILE)
        return

    results = {
        "decreasing": {a: np.empty_like(N_STEPS, dtype=float) for a in ALPHAS},
        "increasing": {a: np.empty_like(N_STEPS, dtype=float) for a in ALPHAS},
        "distance":   np.empty_like(N_STEPS, dtype=float),
    }

    tasks_dec  = list(itertools.product(ALPHAS, N_STEPS))
    tasks_inc  = list(itertools.product(ALPHAS, N_STEPS))
    tasks_dist = list(N_STEPS)
    TOTAL      = len(tasks_dec) + len(tasks_inc) + len(tasks_dist)
    done       = 0

    log.info("Launching %d worker processes (max %d).", TOTAL, MAX_WORKERS)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS,
                             mp_context=CTX) as pool:

        futures = (
            [pool.submit(dec_worker,  t) for t in tasks_dec]  +
            [pool.submit(inc_worker,  t) for t in tasks_inc]  +
            [pool.submit(dist_worker, n) for n in tasks_dist]
        )

        for fut in as_completed(futures):
            kind, alpha, n, tau = fut.result()
            idx = n - 1
            if kind == "dec":
                results["decreasing"][alpha][idx] = tau
            elif kind == "inc":
                results["increasing"][alpha][idx] = tau
            else:
                results["distance"][idx] = tau

            done += 1
            log.info("➀%4d/%d finished (%s α=%s, n=%d)", done, TOTAL, kind, alpha, n)

    with open(OUTFILE, "wb") as fh:
        pickle.dump({"n_steps": N_STEPS, **results}, fh)
    log.info("All %d tasks done – results dumped to %s", TOTAL, OUTFILE)


if __name__ == "__main__":
    main()
