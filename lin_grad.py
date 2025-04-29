#!/usr/bin/env python3
"""
PEPit sweep – **linear η(t)=t+1** only
• BLAS threads capped
• spawn context
• progress logging
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
L, beta, D = 1.0, 1.0, 1.0
N_STEPS    = np.arange(1, 101)
OUTFILE    = "pep_results_lin_grad.pkl"

MAX_WORKERS = min(32, os.cpu_count())
CTX         = mp.get_context("spawn")

# ------------------------------------------------------------------
# 4.  Worker – linear step size only
# ------------------------------------------------------------------
def lin_grad_worker(n):
    """Worst-case ‖∇f(x_n)‖² with η_t = (t+1)/L ."""
    try:
        problem = PEP()
        f       = problem.declare_function(SmoothFunction, L=L)
        x0      = problem.set_initial_point()
        x = z   = x0
        _, f0   = f.oracle(x0)

        for k in range(n):
            y, _   = (1 - beta) * z + beta * x, None
            gy, _  = f.oracle(y)
            z     -= (k + 1) * gy               # linear step
            x      = (1 - 1/(k+1)) * x + (1/(k+1)) * z
            gx, _  = f.oracle(x)
            problem.set_performance_metric(gx ** 2)

        problem.set_initial_condition((f0 - f.oracle(x)[1]) <= D)
        tau = problem.solve(wrapper="cvxpy", verbose=0)
        return n, tau                           # simple tuple

    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        raise

# ------------------------------------------------------------------
# 5.  Driver
# ------------------------------------------------------------------
def main():
    if os.path.exists(OUTFILE):
        log.error("%s already exists – aborting.", OUTFILE)
        return

    lin_grad = np.empty_like(N_STEPS, dtype=float)
    TOTAL    = len(N_STEPS)
    done     = 0

    log.info("Launching %d linear-η tasks with %d workers …",
             TOTAL, MAX_WORKERS)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS,
                             mp_context=CTX) as pool:
        futures = [pool.submit(lin_grad_worker, n) for n in N_STEPS]

        for fut in as_completed(futures):
            n, tau = fut.result()
            lin_grad[n - 1] = tau
            done += 1
            log.info("%3d/%d  n=%d finished", done, TOTAL, n)

    with open(OUTFILE, "wb") as fh:
        pickle.dump({"n_steps": N_STEPS, "lin_grad": lin_grad}, fh)
    log.info("All tasks done – results saved to %s", OUTFILE)

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
