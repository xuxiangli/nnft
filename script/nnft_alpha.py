"""Reproduce the letter-figure data: G^(4) of CosNetFT in d=2 (JAX backend).

Scans (N, alpha, rescale) on a fixed set of n_configs random 4-point
configurations (avg pairwise distance normalized to 1 in the unscaled
configs) at M field samples per setting. Writes one .npz per setting in
`output_dir`, in the format consumed by the letter plotting notebook.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import jax.random as jr

from nnft import CosNetFT, G4_free, IIDSampler, Theory


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
batch_size  = int(1e3)
sample_seed = 42
config_seed = 123
output_dir  = Path("data/nnft_alpha/scan_1e8_float32")


def generate_four_point_configs(d, n_configs, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n_configs, 4, d))
    pts -= pts.mean(axis=1, keepdims=True)
    diffs = pts[:, :, None, :] - pts[:, None, :, :]
    avg_pair = np.linalg.norm(diffs, axis=-1).sum(axis=(1, 2)) / 12.0
    return pts / avg_pair[:, None, None]


def theory_g4(scaled_configs, d, m, Lambda, regulator):
    out = np.empty(scaled_configs.shape[0])
    for ci, pts in enumerate(scaled_configs):
        out[ci] = G4_free(
            pts[0], pts[1], pts[2], pts[3],
            d=d, m=m, Lambda=Lambda, regulator=regulator,
        )
    return out


def run_one(d, N, alpha, Lambda, rescale, configs, M, batch_size, key):
    scaled = configs * rescale
    theory_reg = theory_g4(scaled, d, m, Lambda, regulator)
    x_points = scaled.reshape(-1, d)

    arch = CosNetFT(d_in=d, m=m, alpha=alpha, Lambda=Lambda, regulator=regulator)
    theory = Theory(
        architecture=arch, N=N,
        param_dists=arch.default_dists(),
        normalization="1/sqrt(N)",
    )
    means, errors = theory.correlator(
        x_points, M, key,
        sampler=IIDSampler(),
        n_configs=configs.shape[0],
        batch_size=batch_size,
    )
    return means, errors, theory_reg, scaled


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

        key = jr.fold_in(jr.PRNGKey(sample_seed), i)
        t0 = time.time()
        means, errors, theory_reg, scaled = run_one(
            d, N, alpha, Lambda, rescale, configs, M, batch_size, key,
        )
        dt = time.time() - t0
        np.savez(
            out_path,
            means=np.asarray(means), errors=np.asarray(errors),
            theory_reg=theory_reg, configs=scaled,
        )
        print(
            f"[{i+1}/{len(tasks)}] N={N:3d} alpha={alpha:+d} r={rescale:g}  "
            f"mean ratio = {(np.asarray(means) / theory_reg).mean():.4f}  "
            f"({dt:.1f}s)  -> {fname}"
        )


if __name__ == "__main__":
    main()
