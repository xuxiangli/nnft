"""Smoke checks for nnft package.

1. CosNet 2-point matches analytic kernel at large N.
   For W0 ~ N(0, sigma_W0^2 I_d), b0 ~ U(0, 2 pi), W1 ~ N(0, sigma_W1^2):
       K(x,y) = (sigma_W1^2 / 2) exp(-sigma_W0^2 |x-y|^2 / (2 * d_in))

2. Connected 4-point of CosNet scales as 1/N.

3. CosNet-FT 2-point matches the regulated free-FT kernel
       G(x,y) = int d^d k/(2 pi)^d  f_Lambda(k^2) / (k^2 + m^2) e^{i k . (x - y)}.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from nnft import (
    CosNet,
    CosNetFT,
    G2_free,
    IIDSampler,
    Normal,
    Theory,
    Uniform,
)


def cos_two_point_check():
    print("=== CosNet 2-point vs analytic kernel ===")
    d = 2
    N = 1000
    sigma_W0 = 1.0
    sigma_W1 = 1.0
    n_samples = 10000
    rng = np.random.default_rng(0)

    theory = Theory(
        architecture=CosNet(d_in=d),
        N=N,
        param_dists={
            "W0": Normal(0.0, sigma_W0 / np.sqrt(d)),
            "b0": Uniform(-np.pi, np.pi),
            "W1": Normal(0.0, sigma_W1),
        },
        normalization="1/sqrt(N)",
    )

    x = np.array([0.0, -0.0])
    y = np.array([0.1, 0.0])
    G2, se = theory.correlator([x, y], n_samples, rng, sampler=IIDSampler())
    analytic = (sigma_W1**2 / 2) * np.exp(
        -sigma_W0**2 * np.sum((x - y) ** 2) / (2 * d)
    )
    print(f"  G^(2)(x,y) = {G2:.5f} +/- {se:.5f}")
    print(f"  analytic   = {analytic:.5f}")
    print(f"  |diff|/se  = {abs(G2 - analytic) / se:.2f}")


def cos_connected_4pt_scaling():
    print("\n=== CosNet connected 4-point: 1/N scaling (coincident pts) ===")
    d = 2
    sigma_W0 = 1.0
    sigma_W1 = 1.0
    n_samples = 200000
    rng = np.random.default_rng(2)

    x = np.array([0.3, -0.1])
    pts = [x, x, x, x]

    for N in [10, 30, 50, 100]:
        theory = Theory(
            architecture=CosNet(d_in=d),
            N=N,
            param_dists={
                "W0": Normal(0.0, sigma_W0 / np.sqrt(d)),
                "b0": Uniform(-np.pi, np.pi),
                "W1": Normal(0.0, sigma_W1),
            },
            normalization="1/sqrt(N)",
        )
        kappa4, se = theory.correlator(
            pts, n_samples, rng, connected=True, bootstrap=100
        )
        print(f"  N={N:>4d}  kappa_4 = {kappa4:+.5f} +/- {se:.5f}   "
              f"N*kappa_4 = {N * kappa4:+.4f}")


def cosnet_ft_two_point_check(d=1, regulator="hard", seed=3):
    print(
        f"\n=== CosNet-FT 2-point vs regulated free-FT kernel "
        f"(d={d}, {regulator}) ==="
    )
    m = 1.0
    alpha = 0.0
    Lambda = 5.0
    N = 2000
    n_samples = 20000
    rng = np.random.default_rng(seed)

    arch = CosNetFT(d_in=d, m=m, alpha=alpha, Lambda=Lambda, regulator=regulator)
    theory = Theory(
        architecture=arch,
        N=N,
        param_dists=arch.default_dists(),
        normalization="1/sqrt(N)",
    )

    x0 = np.zeros(d)
    for sep in [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]:
        y = np.zeros(d)
        y[0] = sep
        G2_mc, se = theory.correlator([x0, y], n_samples, rng, sampler=IIDSampler())
        G2_an = G2_free(x0, y, d=d, m=m, Lambda=Lambda, regulator=regulator)
        print(
            f"  r={sep:>4.2f}  G2_MC = {G2_mc:+.5f} +/- {se:.5f}   "
            f"G2_an = {G2_an:+.5f}   |diff|/se = {abs(G2_mc - G2_an) / se:.2f}"
        )


if __name__ == "__main__":
    # cos_two_point_check()
    # cos_connected_4pt_scaling()
    for d in (1, 2, 3):
        cosnet_ft_two_point_check(d=d, regulator="hard", seed=3 + d)
        cosnet_ft_two_point_check(d=d, regulator="gaussian", seed=13 + d)
