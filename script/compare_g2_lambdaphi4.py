"""Compare 2-point correlator G^(2)(r) in lambda phi^4: NNFT vs analytics (JAX).

NNFT methods compared:
  - method A: free-theory sampling reweighted by exp(-S_int)
  - method B: Metropolis-Hastings on the modified PDF
  - method C: Hamiltonian Monte Carlo (blackjax) on the modified PDF
  - method D: MALA (blackjax) on the modified PDF

Analytical curves (tree + one-loop, with and without IR vertex regulator)
come from analytics.AnalFree / AnalPhi4 (NumPy/SciPy).

Saves data to data/lambda_phi4/compare_g2_l{lambda}.npz and a plot to
.../compare_g2_l{lambda}.png. Lattice overlay is loaded if
data/lambda_phi4/lattice_g2_l{lambda}.npz is present.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import jax.random as jr

from nnft import (
    AnalFree,
    AnalPhi4,
    CosNetFT,
    HMCSampler,
    LambdaPhi4,
    MALASampler,
    MetropolisHastingsSampler,
    Theory,
)


# ---------- parameters ---------------------------------------------------

d         = 2
m         = 1.0
lambda_   = 6.0
Lambda    = 100.0
regulator = "gaussian"
alpha     = 0
L         = 10.0
N         = 50

# Sample counts
M_reweight = int(1e6)
M_mh       = int(1e4)
M_hmc      = int(1e4)
M_mala     = int(5e3)
batch_size = 500

# Sampler controls
mh_burn_in = 500
mh_step    = 1e-3
mh_thin    = 1
hmc_burn_in = 500
hmc_step = 5e-4
hmc_n_leapfrog = 5
hmc_thin = 1
mala_burn_in = 500
mala_step = 1e-4
mala_thin = 1

# Interaction action method (used by MCMC samplers via Theory.log_density)
action_method = "trans_sym"
reweight_action_method = "real_space_mc"
reweight_M_x = 1024

r_grid = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0])
seed = 0
output_dir = Path("data/lambda_phi4")


_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--lambda", dest="_lambda_cli", type=float, default=None)
_p.add_argument("--suffix", dest="_suffix_cli", type=str, default=None)
_p.add_argument("--quick", action="store_true",
                help="reduced sample counts for smoke testing")
_cli, _ = _p.parse_known_args()
if _cli._lambda_cli is not None:
    lambda_ = _cli._lambda_cli
if _cli.quick:
    M_reweight = int(1e4)
    M_mh = int(1e3)
    M_hmc = int(500)
    M_mala = int(500)
_suffix = _cli._suffix_cli if _cli._suffix_cli is not None else f"_l{lambda_}"


# ---------- analytical curves -------------------------------------------
def _sym_pair(r, d):
    x1 = np.r_[+r / 2.0, np.zeros(d - 1)]
    x2 = np.r_[-r / 2.0, np.zeros(d - 1)]
    return x1, x2


def analytical_curves(r_grid):
    free = AnalFree(d=d, m=m, Lambda=Lambda, regulator=regulator)
    phi4_no_ir = AnalPhi4(free, lambda_=lambda_)
    phi4_ir = AnalPhi4(free, lambda_=lambda_, L=L)
    n_r = len(r_grid)
    Delta_uv = np.empty(n_r)
    G2_no_ir = np.empty(n_r)
    G2_ir = np.empty(n_r)
    for i, r in enumerate(r_grid):
        Delta_uv[i] = free.G2(r, method="asymptotic_large_r"
                              if r > 0.05 else "asymptotic_small_r")
        x1, x2 = _sym_pair(r, d)
        G2_no_ir[i] = phi4_no_ir.G2_one_loop(x1, x2, method="schwinger_t")
        G2_ir[i] = phi4_ir.G2_one_loop(x1, x2, method="schwinger_t1t2")
    return Delta_uv, G2_no_ir, G2_ir


def x_points_for_grid(r_grid, d):
    pts = np.zeros((len(r_grid), 2, d))
    pts[:, 0, 0] = +r_grid / 2.0
    pts[:, 1, 0] = -r_grid / 2.0
    return pts.reshape(-1, d)


# ---------- NNFT methods -------------------------------------------------
def _make_theory_with_interaction(phi4, method):
    arch = CosNetFT(d_in=d, m=m, alpha=alpha, Lambda=Lambda, regulator=regulator)
    kwargs = {}
    if method == "trans_sym":
        kwargs = {}
    elif method == "real_space_hermite":
        kwargs = {"n_hermite": 6}
    return Theory(
        arch, N=N, param_dists=arch.default_dists(),
        interaction=phi4, action_method=method, action_kwargs=kwargs,
    )


def nnft_reweighted(phi4, r_grid, M, key):
    arch = CosNetFT(d_in=d, m=m, alpha=alpha, Lambda=Lambda, regulator=regulator)
    th = Theory(arch, N=N, param_dists=arch.default_dists())  # no interaction
    x_points = x_points_for_grid(r_grid, d)
    return th.correlator_reweighted(
        x_points, M, key,
        interaction=phi4, n_configs=len(r_grid), batch_size=batch_size,
        action_method=reweight_action_method,
        action_kwargs={"M_x": reweight_M_x},
    )


def nnft_mcmc(sampler, phi4, r_grid, M, key, method):
    th = _make_theory_with_interaction(phi4, method)
    x_points = x_points_for_grid(r_grid, d)
    means, errs = th.correlator(
        x_points, M, key,
        sampler=sampler, n_configs=len(r_grid),
        batch_size=min(batch_size, M),
    )
    return means, errs, sampler.last_acceptance


# ---------- main --------------------------------------------------------
def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    phi4 = LambdaPhi4(lambda_, L, "gaussian")

    print(f"d={d}  m={m}  Lambda={Lambda}  regulator={regulator}  alpha={alpha}")
    print(f"L={L}  lambda={lambda_}  N={N}")
    print(f"M_reweight={M_reweight:.1e}  M_mh={M_mh}  M_hmc={M_hmc}  M_mala={M_mala}")
    print(f"action_method={action_method} (MCMC)  reweight={reweight_action_method}")
    print(f"r_grid = {r_grid}")
    print()

    # analytics
    print("computing analytical curves...")
    t0 = time.time()
    Delta_uv, G2_loop_no_ir, G2_loop_ir = analytical_curves(r_grid)
    print(f"  done ({time.time() - t0:.1f}s)")

    # method A: reweighting
    print(f"NNFT A: reweighting (M={M_reweight:.1e})...")
    t0 = time.time()
    rw_mean, rw_err = nnft_reweighted(phi4, r_grid, M_reweight, jr.PRNGKey(seed))
    print(f"  done ({time.time() - t0:.1f}s)")

    # method B: MH
    print(f"NNFT B: MH (steps={M_mh}, burn={mh_burn_in})...")
    t0 = time.time()
    mh = MetropolisHastingsSampler(
        step_size=mh_step, n_warmup=mh_burn_in, thin=mh_thin
    )
    mh_mean, mh_err, mh_acc = nnft_mcmc(
        mh, phi4, r_grid, M_mh, jr.PRNGKey(seed + 100), action_method
    )
    print(f"  done ({time.time() - t0:.1f}s)  acc={mh_acc:.3f}")

    # method C: HMC
    print(f"NNFT C: HMC (steps={M_hmc}, burn={hmc_burn_in})...")
    t0 = time.time()
    hmc = HMCSampler(
        step_size=hmc_step,
        num_integration_steps=hmc_n_leapfrog,
        n_warmup=hmc_burn_in,
        thin=hmc_thin,
    )
    hmc_mean, hmc_err, hmc_acc = nnft_mcmc(
        hmc, phi4, r_grid, M_hmc, jr.PRNGKey(seed + 200), action_method
    )
    print(f"  done ({time.time() - t0:.1f}s)  acc={hmc_acc:.3f}")

    # method D: MALA
    print(f"NNFT D: MALA (steps={M_mala}, burn={mala_burn_in})...")
    t0 = time.time()
    mala = MALASampler(step_size=mala_step, n_warmup=mala_burn_in, thin=mala_thin)
    mala_mean, mala_err, mala_acc = nnft_mcmc(
        mala, phi4, r_grid, M_mala, jr.PRNGKey(seed + 300), action_method
    )
    print(f"  done ({time.time() - t0:.1f}s)  acc={mala_acc:.3f}")

    np.savez(
        output_dir / f"compare_g2{_suffix}.npz",
        r_grid=r_grid,
        Delta_uv=Delta_uv,
        G2_one_loop_no_ir=G2_loop_no_ir,
        G2_one_loop_ir=G2_loop_ir,
        rw_mean=np.asarray(rw_mean), rw_err=np.asarray(rw_err),
        mh_mean=np.asarray(mh_mean), mh_err=np.asarray(mh_err),
        hmc_mean=np.asarray(hmc_mean), hmc_err=np.asarray(hmc_err),
        mala_mean=np.asarray(mala_mean), mala_err=np.asarray(mala_err),
        d=d, m=m, Lambda=Lambda, regulator=regulator, alpha=alpha,
        L=L, lambda_=lambda_, N=N,
        M_reweight=M_reweight, M_mh=M_mh, M_hmc=M_hmc, M_mala=M_mala,
        mh_acceptance=mh_acc, hmc_acceptance=hmc_acc, mala_acceptance=mala_acc,
        action_method=action_method,
        reweight_action_method=reweight_action_method,
    )

    header = (
        f"{'r':>5}  {'Delta_uv':>10}  {'1-loop':>10}  {'1-loop IR':>10}  "
        f"{'reweight':>14}  {'MH':>14}  {'HMC':>14}  {'MALA':>14}"
    )
    print()
    print(header)
    print("-" * len(header))
    for i, r in enumerate(r_grid):
        print(
            f"{r:5.2f}  {Delta_uv[i]:10.4f}  {G2_loop_no_ir[i]:10.4f}  "
            f"{G2_loop_ir[i]:10.4f}  "
            f"{rw_mean[i]:7.4f}+/-{rw_err[i]:5.4f}  "
            f"{mh_mean[i]:7.4f}+/-{mh_err[i]:5.4f}  "
            f"{hmc_mean[i]:7.4f}+/-{hmc_err[i]:5.4f}  "
            f"{mala_mean[i]:7.4f}+/-{mala_err[i]:5.4f}"
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(r_grid, Delta_uv, "k-", label="tree, UV reg")
        ax.plot(r_grid, G2_loop_no_ir, "C0-", label="1-loop, no IR")
        ax.plot(r_grid, G2_loop_ir, "C1-", label="1-loop, Gaussian IR")
        ax.errorbar(r_grid, rw_mean, yerr=rw_err, fmt="C2o",
                    label=f"NNFT reweight (M={M_reweight:.1e})", capsize=3)
        ax.errorbar(r_grid, mh_mean, yerr=mh_err, fmt="C3s",
                    label=f"NNFT MH (M={M_mh}, acc={mh_acc:.2f})", capsize=3)
        ax.errorbar(r_grid, hmc_mean, yerr=hmc_err, fmt="C4^",
                    label=f"NNFT HMC (M={M_hmc}, acc={hmc_acc:.2f})", capsize=3)
        ax.errorbar(r_grid, mala_mean, yerr=mala_err, fmt="C5v",
                    label=f"NNFT MALA (M={M_mala}, acc={mala_acc:.2f})", capsize=3)

        lattice_path = output_dir / f"lattice_g2{_suffix}.npz"
        if lattice_path.exists():
            lz = np.load(lattice_path, allow_pickle=True)
            ax.errorbar(lz["r_grid"], lz["lat_mean"], yerr=lz["lat_err"],
                        fmt="C6D", label="lattice", capsize=3)

        ax.set_xlabel("r")
        ax.set_ylabel(r"$G^{(2)}(r)$")
        ax.set_title(
            rf"$\lambda\phi^4$: $d={d}$, $m={m}$, $\Lambda={Lambda}$, "
            rf"$L={L}$, $\lambda={lambda_}$, $N={N}$"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_png = output_dir / f"compare_g2{_suffix}.png"
        fig.savefig(out_png, dpi=140)
        print(f"\nplot saved to {out_png}")
    except ImportError:
        print("\nmatplotlib not available; skipped plotting.")


if __name__ == "__main__":
    main()
