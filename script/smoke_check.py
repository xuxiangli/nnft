"""Smoke checks for the JAX-backed nnft package.

1. CosNet 2-point matches analytic kernel at large N (IIDSampler).
       For W ~ N(0, sigma_W^2 I_d), b ~ U(0, 2 pi):
       K(x, y) = (1/2) exp(-sigma_W^2 |x-y|^2 / 2)

2. DenseTanh connected 4-point scales as 1/N (IIDSampler).

3. MetropolisHastingsSampler reproduces the IIDSampler 2-point on the
   factorized prior (sanity check that the chain mixes correctly).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from nnft import (
    CosNet,
    DenseTanh,
    MetropolisHastingsSampler,
    Normal,
    Theory,
    Uniform,
)


def cos_two_point_check():
    print("=== CosNet 2-point vs analytic kernel (IID) ===")
    d = 2
    sigma_W = 1.0
    N = 2000
    n_samples = 4000

    theory = Theory(
        architecture=CosNet(d_in=d),
        N=N,
        param_dists={"W": Normal(0.0, sigma_W), "b": Uniform(0.0, 2 * jnp.pi)},
    )

    x = jnp.array([0.3, -0.1])
    y = jnp.array([0.5, 0.4])
    G2, se = theory.correlator(jnp.stack([x, y]), n_samples, jr.PRNGKey(0))
    analytic = 0.5 * float(jnp.exp(-sigma_W**2 * jnp.sum((x - y) ** 2) / 2))
    print(f"  G^(2)(x,y) = {G2:.5f} +/- {se:.5f}")
    print(f"  analytic   = {analytic:.5f}")
    print(f"  |diff|/se  = {abs(G2 - analytic) / se:.2f}")


def tanh_connected_4pt_scaling():
    print("\n=== DenseTanh connected 4-point: 1/N scaling (IID, coincident pts) ===")
    n_samples = 50_000

    pts = jnp.array([[0.5], [0.5], [0.5], [0.5]])
    for N in [10, 30, 100]:
        theory = Theory(
            architecture=DenseTanh(d_in=1),
            N=N,
            param_dists={"W": Normal(0.0, 1.0), "b": Normal(0.0, 1.0)},
        )
        kappa4, se = theory.correlator(
            pts, n_samples, jr.PRNGKey(N), connected=True, bootstrap=80
        )
        print(
            f"  N={N:>4d}  kappa_4 = {kappa4:+.5f} +/- {se:.5f}   "
            f"N*kappa_4 = {N * kappa4:+.4f}"
        )


def mh_matches_iid_on_factorized_prior():
    print("\n=== Metropolis-Hastings on factorized prior matches IID ===")
    d = 1
    sigma_W = 1.0
    N = 30  # total proposal dim = N * (d + 1) = 60
    n_samples = 4000

    theory = Theory(
        architecture=DenseTanh(d_in=d),
        N=N,
        param_dists={"W": Normal(0.0, sigma_W), "b": Normal(0.0, 1.0)},
    )

    x = jnp.array([0.2])
    y = jnp.array([0.6])
    pts = jnp.stack([x, y])

    G2_iid, se_iid = theory.correlator(pts, n_samples, jr.PRNGKey(1))

    # Step ~ 2.4 / sqrt(dim) per Roberts-Gelman-Gilks; dim=60 -> ~0.31.
    # Take a slightly smaller step + thinning for safety.
    mh = MetropolisHastingsSampler(step_size=0.2, n_warmup=4000, thin=10)
    G2_mh, se_mh = theory.correlator(pts, n_samples, jr.PRNGKey(2), sampler=mh)

    print(f"  IID:  G^(2) = {G2_iid:+.4f} +/- {se_iid:.4f}")
    print(f"  MH :  G^(2) = {G2_mh:+.4f} +/- {se_mh:.4f}   "
          f"(acceptance = {mh.last_acceptance:.2f})")
    diff_se = float(np.hypot(se_iid, se_mh))
    print(f"  |IID - MH| / sqrt(se_iid^2 + se_mh^2) = "
          f"{abs(G2_iid - G2_mh) / diff_se:.2f}")


if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")
    cos_two_point_check()
    tanh_connected_4pt_scaling()
    mh_matches_iid_on_factorized_prior()
