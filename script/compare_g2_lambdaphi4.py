"""Compare 2-point correlator G^(2)(r) in lambda phi^4 between NNFT and analytics.

Compares six curves over a 1D radial grid r:
  - tree-level free propagator with UV regulator
  - tree-level free propagator without UV regulator (analytic Bessel form)
  - one-loop tadpole, no IR vertex regulator (translation invariant)
  - one-loop tadpole, with Gaussian IR vertex regulator
  - NNFT method A: free-theory sampling reweighted by exp(-S_int)
  - NNFT method B: Metropolis-Hastings on the modified PDF

Saves data to data/lambda_phi4/compare_g2.npz and a plot to .../compare_g2.png.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from nnft import (
    CosNetFT,
    G2_lambda_phi4_one_loop_ir,
    G2_lambda_phi4_one_loop_no_ir,
    LambdaPhi4,
    MetropolisHastingsSampler,
    Theory,
    propagator,
)


# ---------- parameters ---------------------------------------------------
d           = 2
m           = 1.0
Lambda      = 100.0
regulator   = "gaussian"
alpha       = 0
L           = 10.0
lambda_     = 6.0
N           = 50
M_reweight  = int(2e6)
M_mh        = int(5e4)         # MH samples after burn-in (each is a sweep)
M_x         = 200              # MC quadrature points for S_int
batch_size  = 500              # field-sample chunk for memory control
mh_burn_in  = 500
mh_thin     = 1
mh_W0_step  = 1.0              # increased from 0.3 to lower acceptance, improve mixing
mh_b0_step  = np.pi
n_r         = 9
r_grid      = np.linspace(0.1, 3.0, n_r)
seed        = 0
cache_analytics = True         # reuse analytical curves from prior .npz when params match

output_dir  = Path("data/lambda_phi4")


# ---------- analytical curves ---------------------------------------------
def analytical_curves(r_grid):
    Delta_uv = np.array([propagator(r, d, m, Lambda, regulator) for r in r_grid])
    Delta_no_uv = np.array(
        [propagator(r, d, m, regulator="none") for r in r_grid]
    )
    G2_one_loop_no_ir = np.array([
        G2_lambda_phi4_one_loop_no_ir(
            np.zeros(d), np.r_[r, np.zeros(d - 1)],
            d, m, lambda_, Lambda, regulator,
        )
        for r in r_grid
    ])
    G2_one_loop_ir = np.array([
        G2_lambda_phi4_one_loop_ir(
            np.zeros(d), np.r_[r, np.zeros(d - 1)],
            d, m, lambda_, L, Lambda, regulator,
            n_hermite=24,
        )
        for r in r_grid
    ])
    return Delta_uv, Delta_no_uv, G2_one_loop_no_ir, G2_one_loop_ir


# ---------- NNFT methods --------------------------------------------------
def x_points_for_grid(r_grid, d):
    """Construct evaluation points for each separation in ``r_grid``.

    For every ``r`` this returns the pair 
    ``x1 = (0, 0, ..., 0)`` and ``x2 = (r, 0, ..., 0)``
    in a flattened array of shape ``(2 * len(r_grid), d)``.
    """
    pts = np.zeros((len(r_grid), 2, d))
    pts[:, 1, 0] = r_grid
    return pts.reshape(-1, d)


def nnft_reweighted(theory, phi4, r_grid, M, rng):
    x_points = x_points_for_grid(r_grid, theory.architecture.d_in)
    return theory.correlator_reweighted(
        x_points, M, rng,
        interaction=phi4,
        n_configs=len(r_grid),
        batch_size=batch_size,
        action_method="real_space_mc",
        action_kwargs={"M_x": M_x},
    )


def nnft_mh(theory, phi4, r_grid, M, rng):
    proposals = {
        "W0": ("normal", mh_W0_step),
        "b0": ("uniform_wrap", mh_b0_step),
        "W1": ("none", 0.0),
    }
    mh = MetropolisHastingsSampler(
        phi4, proposals,
        proposal_mode="single",
        burn_in=mh_burn_in, thin=mh_thin,
        action_method="real_space_mc",
        action_kwargs={"M_x": M_x},
    )
    x_points = x_points_for_grid(r_grid, theory.architecture.d_in)
    means, errs = theory.correlator(
        x_points, M, rng,
        sampler=mh, n_configs=len(r_grid),
    )
    return means, errs, mh.acceptance_rate


# ---------- main ----------------------------------------------------------
def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    arch = CosNetFT(d_in=d, m=m, alpha=alpha, Lambda=Lambda, regulator=regulator)
    th = Theory(arch, N=N, param_dists=arch.default_dists())
    phi4 = LambdaPhi4(lambda_, L, "gaussian")

    print(f"d={d}  m={m}  Lambda={Lambda}  regulator={regulator}  alpha={alpha}")
    print(f"L={L}  lambda={lambda_}  N={N}")
    print(f"M_reweight={M_reweight:.1e}  M_mh={M_mh}  M_x={M_x}")
    print(f"r_grid = {r_grid}")
    print()

    cache_path = output_dir / "compare_g2.npz"
    cached = None
    if cache_analytics and cache_path.exists():
        try:
            z = np.load(cache_path, allow_pickle=True)
            same = (
                z["d"] == d and z["m"] == m and z["Lambda"] == Lambda
                and str(z["regulator"]) == regulator and z["alpha"] == alpha
                and z["L"] == L and z["lambda_"] == lambda_
                and len(z["r_grid"]) == len(r_grid)
                and np.allclose(z["r_grid"], r_grid)
            )
            if same:
                cached = z
        except Exception:
            cached = None

    t_analytic = 0.0
    if cached is not None:
        print("reusing cached analytical curves from previous run.")
        Delta_uv = cached["Delta_uv"]
        Delta_no_uv = cached["Delta_no_uv"]
        G2_loop_no_ir = cached["G2_one_loop_no_ir"]
        G2_loop_ir = cached["G2_one_loop_ir"]
    else:
        print("computing analytical curves...")
        t0 = time.time()
        Delta_uv, Delta_no_uv, G2_loop_no_ir, G2_loop_ir = analytical_curves(r_grid)
        t_analytic = time.time() - t0
        print(f"  done ({t_analytic:.1f}s)")

    print(f"NNFT method A: free-theory reweighting (M={M_reweight:.1e})...")
    t0 = time.time()
    rng = np.random.default_rng(seed)
    rw_mean, rw_err = nnft_reweighted(th, phi4, r_grid, M_reweight, rng)
    t_reweight = time.time() - t0
    print(f"  done ({t_reweight:.1f}s)")

    print(f"NNFT method B: Metropolis-Hastings (sweeps={M_mh}, burn={mh_burn_in})...")
    t0 = time.time()
    rng = np.random.default_rng(seed + 100)
    mh_mean, mh_err, acc = nnft_mh(th, phi4, r_grid, M_mh, rng)
    t_mh = time.time() - t0
    print(f"  done ({t_mh:.1f}s)  acceptance = {acc:.3f}")

    np.savez(
        output_dir / "compare_g2.npz",
        r_grid=r_grid,
        Delta_uv=Delta_uv,
        Delta_no_uv=Delta_no_uv,
        G2_one_loop_no_ir=G2_loop_no_ir,
        G2_one_loop_ir=G2_loop_ir,
        rw_mean=rw_mean, rw_err=rw_err,
        mh_mean=mh_mean, mh_err=mh_err,
        d=d, m=m, Lambda=Lambda, regulator=regulator, alpha=alpha,
        L=L, lambda_=lambda_, N=N,
        M_reweight=M_reweight, M_mh=M_mh, M_x=M_x,
    )

    print()
    print("=== timing summary ===")
    if t_analytic > 0:
        print(f"  analytics (4 curves, {len(r_grid)} r values): {t_analytic:.1f}s")
    print(f"  NNFT reweight (M={M_reweight:.1e}, M_x={M_x}):    {t_reweight:.1f}s "
          f"-> {t_reweight / M_reweight * 1e6:.2f} us / sample")
    print(f"  NNFT MH       (sweeps={M_mh}, burn={mh_burn_in}): {t_mh:.1f}s "
          f"-> {t_mh / M_mh * 1e3:.2f} ms / sweep,  acc={acc:.3f}")
    print()
    header = (
        f"{'r':>5}  {'Delta_uv':>10}  {'1-loop':>10}  {'1-loop IR':>10}  "
        f"{'reweight':>14}  {'MH':>14}"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(r_grid):
        print(
            f"{r:5.2f}  {Delta_uv[i]:10.4f}  {G2_loop_no_ir[i]:10.4f}  "
            f"{G2_loop_ir[i]:10.4f}  "
            f"{rw_mean[i]:7.4f}+/-{rw_err[i]:5.4f}  "
            f"{mh_mean[i]:7.4f}+/-{mh_err[i]:5.4f}"
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(r_grid, Delta_uv, "k-", label=r"tree, UV reg")
        ax.plot(r_grid, Delta_no_uv, "k--", label=r"tree, no UV reg")
        ax.plot(r_grid, G2_loop_no_ir, "C0-", label=r"1-loop, no IR")
        ax.plot(r_grid, G2_loop_ir, "C1-", label=r"1-loop, Gaussian IR")
        ax.errorbar(r_grid, rw_mean, yerr=rw_err, fmt="C2o",
                    label=f"NNFT reweight (M={M_reweight:.1e})", capsize=3)
        ax.errorbar(r_grid, mh_mean, yerr=mh_err, fmt="C3s",
                    label=f"NNFT MH (sweeps={M_mh})", capsize=3)
        ax.set_xlabel("r")
        ax.set_ylabel(r"$G^{(2)}(r)$")
        ax.set_title(
            rf"$\lambda\phi^4$: $d={d}$, $m={m}$, $\Lambda={Lambda}$, "
            rf"$L={L}$, $\lambda={lambda_}$, $N={N}$"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_png = output_dir / "compare_g2.png"
        fig.savefig(out_png, dpi=140)
        print(f"\nplot saved to {out_png}")
    except ImportError:
        print("\nmatplotlib not available; skipped plotting.")


if __name__ == "__main__":
    main()
