"""Smoke checks for nnft package.

1. CosNet 2-point matches analytic kernel at large N.
   For W ~ N(0, sigma_W^2 I_d), b ~ U(0, 2 pi):
       K(x,y) = (1/2) exp(-sigma_W^2 |x-y|^2 / 2)

2. Connected 4-point of DenseTanh scales as 1/N.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from nnft import CosNet, DenseTanh, IIDSampler, Normal, Theory, Uniform


def cos_two_point_check():
    print("=== CosNet 2-point vs analytic kernel ===")
    d = 2
    sigma_W = 1.0
    N = 2000
    n_samples = 4000
    rng = np.random.default_rng(0)

    theory = Theory(
        architecture=CosNet(d_in=d),
        N=N,
        param_dists={
            "W": Normal(0.0, sigma_W),
            "b": Uniform(0.0, 2 * np.pi),
        },
        normalization="1/sqrt(N)",
    )

    x = np.array([0.3, -0.1])
    y = np.array([0.5, 0.4])
    G2, se = theory.correlator([x, y], n_samples, rng, sampler=IIDSampler())
    analytic = 0.5 * np.exp(-sigma_W**2 * np.sum((x - y) ** 2) / 2)
    print(f"  G^(2)(x,y) = {G2:.5f} +/- {se:.5f}")
    print(f"  analytic   = {analytic:.5f}")
    print(f"  |diff|/se  = {abs(G2 - analytic) / se:.2f}")


def tanh_connected_4pt_scaling():
    print("\n=== DenseTanh connected 4-point: 1/N scaling (coincident pts) ===")
    d = 1
    n_samples = 200000
    rng = np.random.default_rng(1)

    x = np.array([0.5])
    pts = [x, x, x, x]  # coincident points: kappa_4 = <phi^4>_c

    for N in [10, 30, 100]:
        theory = Theory(
            architecture=DenseTanh(d_in=d),
            N=N,
            param_dists={"W": Normal(0.0, 1.0), "b": Normal(0.0, 1.0)},
            normalization="1/sqrt(N)",
        )
        kappa4, se = theory.correlator(
            pts, n_samples, rng, connected=True, bootstrap=100
        )
        print(f"  N={N:>4d}  kappa_4 = {kappa4:+.5f} +/- {se:.5f}   "
              f"N*kappa_4 = {N * kappa4:+.4f}")


if __name__ == "__main__":
    cos_two_point_check()
    tanh_connected_4pt_scaling()
