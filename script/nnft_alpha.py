"""Reproduce the letter-figure data: G^(4) of CosNetFT in d=2.

Scans (N, alpha, rescale) on a fixed set of n_configs=50 random 4-point
configurations (avg pairwise distance normalized to 1 in the unscaled
configs) at M field samples per setting. Writes one .npz per setting in
`output_dir`, in the format consumed by ~/Code/nnft-alpha/letter_plot.ipynb.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from nnft import (
    CosNetFT,
    G4_free,
    IIDSampler,
    Theory,
)


# --- letter-figure parameters ---------------------------------------------
d           = 2
m           = 1.0
Lambda      = 100.0
regulator   = "gaussian"
N_values    = [30, 50, 100]
alphas      = [-1, 0, 1]
rescales    = [0.3, 1.0, 3.0]
n_configs   = 5
M           = int(1e8)
M_str       = "1e8"
batch_size  = 1e3
dtype       = np.float32
sample_seed = 42
config_seed = 123
output_dir  = Path("data/nnft_alpha/scan_1e8_float32")


def generate_four_point_configs(d, n_configs, seed=0):
    """`n_configs` random 4-point configs, each with avg pairwise dist = 1."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n_configs, 4, d))
    pts -= pts.mean(axis=1, keepdims=True)
    diffs = pts[:, :, None, :] - pts[:, None, :, :]
    # 12 ordered pairs per config (6 unordered); divide by 12 -> avg of all
    avg_pair = np.linalg.norm(diffs, axis=-1).sum(axis=(1, 2)) / 12.0
    return pts / avg_pair[:, None, None]


def theory_g4(scaled_configs, d, m, Lambda, regulator):
    """G4_free for each (4, d) config."""
    out = np.empty(scaled_configs.shape[0])
    for ci, pts in enumerate(scaled_configs):
        out[ci] = G4_free(
            pts[0], pts[1], pts[2], pts[3],
            d=d, m=m, Lambda=Lambda, regulator=regulator,
        )
    return out


def run_one(d, N, alpha, Lambda, rescale, configs, M, batch_size, seed,
            dtype=np.float64):
    """One (N, alpha, rescale) task: returns (means, errors, theory_reg, scaled_configs).

    Original implementation: draws a fresh M-sample ensemble per rescale.
    """
    scaled = configs * rescale                                 # (n_configs, 4, d)
    theory_reg = theory_g4(scaled, d, m, Lambda, regulator)    # (n_configs,)
    x_points = scaled.reshape(-1, d)                           # (n_configs * 4, d)

    arch = CosNetFT(d_in=d, m=m, alpha=alpha, Lambda=Lambda, regulator=regulator)
    theory = Theory(
        architecture=arch,
        N=N,
        param_dists=arch.default_dists(),
        normalization="1/sqrt(N)",
        dtype=dtype,
    )
    rng = np.random.default_rng(seed)
    means, errors = theory.correlator(
        x_points, M, rng,
        sampler=IIDSampler(),
        n_configs=configs.shape[0],
        batch_size=batch_size,
    )
    return means, errors, theory_reg, scaled


def run_grouped(d, N, alpha, Lambda, rescales, configs, M, batch_size, seed,
                dtype=np.float64):
    """One (N, alpha) task across all rescales — sharing one parameter ensemble.

    Improvement over `run_one`: only draws the M-sample ensemble once per
    (N, alpha) and evaluates phi at all rescales' query points in a single
    correlator call. Saves the per-rescale sampling cost.

    Returns:
        means_by_r:    (n_rescales, n_configs)
        errors_by_r:   (n_rescales, n_configs)
        theory_by_r:   (n_rescales, n_configs)
        scaled_by_r:   (n_rescales, n_configs, 4, d)
    """
    n_r = len(rescales)
    n_c = configs.shape[0]

    # (n_rescales, n_configs, 4, d) — one config block per rescale
    scaled_by_r = np.stack([configs * r for r in rescales], axis=0)
    theory_by_r = np.stack(
        [theory_g4(scaled_by_r[ri], d, m, Lambda, regulator) for ri in range(n_r)],
        axis=0,
    )
    # flatten so the correlator sees n_r * n_c independent 4-point groups
    x_points = scaled_by_r.reshape(n_r * n_c * 4, d)

    arch = CosNetFT(d_in=d, m=m, alpha=alpha, Lambda=Lambda, regulator=regulator)
    theory = Theory(
        architecture=arch,
        N=N,
        param_dists=arch.default_dists(),
        normalization="1/sqrt(N)",
        dtype=dtype,
    )
    rng = np.random.default_rng(seed)
    means, errors = theory.correlator(
        x_points, M, rng,
        sampler=IIDSampler(),
        n_configs=n_r * n_c,
        batch_size=batch_size,
    )
    return (
        means.reshape(n_r, n_c),
        errors.reshape(n_r, n_c),
        theory_by_r,
        scaled_by_r,
    )


def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = generate_four_point_configs(d, n_configs, seed=config_seed)

    tasks = [(N, alpha, rescale)
             for N in N_values for alpha in alphas for rescale in rescales]
    for i, (N, alpha, rescale) in enumerate(tasks):
        fname = (
            f"G4_d{d}_N{N}_L{Lambda:g}_a{alpha:g}_r{rescale:g}_M{M_str}.npz"
        )
        out_path = output_dir / fname
        if out_path.exists():
            print(f"[{i+1}/{len(tasks)}] skip {fname} (exists)")
            continue

        seed = sample_seed
        if seed is None:
            seed = (hash((N, alpha, rescale)) ^ 0xA1B2C3) & ((1 << 63) - 1)
        t0 = time.time()
        means, errors, theory_reg, scaled = run_one(
            d, N, alpha, Lambda, rescale, configs, M, batch_size, seed,
            dtype=dtype,
        )
        dt = time.time() - t0
        np.savez(
            out_path,
            means=means, errors=errors,
            theory_reg=theory_reg, configs=scaled,
        )
        print(
            f"[{i+1}/{len(tasks)}] N={N:3d} alpha={alpha:+d} r={rescale:g}  "
            f"mean ratio = {(means / theory_reg).mean():.4f}  "
            f"({dt:.1f}s)  -> {fname}"
        )


if __name__ == "__main__":
    main()
