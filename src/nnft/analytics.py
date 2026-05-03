"""Analytic helpers for the field-theory NN architectures.

Houses closed-form / special-function expressions that are independent of any
particular sampler or architecture: UV regulators, the normalization constant
Omega_alpha, and the target free-FT 2-point kernel used for verification.
"""

import numpy as np
from scipy import integrate, special

_REGULATORS = ("hard", "gaussian", "none")


def f_Lambda(k_sq, Lambda, regulator):
    """UV regulator f_Lambda(k^2). Vectorized over k_sq."""
    k_sq = np.asarray(k_sq, dtype=float)
    if regulator == "hard":
        return np.where(k_sq <= Lambda * Lambda, 1.0, 0.0)
    if regulator == "gaussian":
        return np.exp(-k_sq / (2.0 * Lambda * Lambda))
    if regulator == "none":
        return np.ones_like(k_sq)
    raise ValueError(f"regulator must be one of {_REGULATORS}, got {regulator!r}")


def omega_alpha(d, m, alpha, Lambda, regulator):
    """Closed-form Omega_alpha = int d^d k/(2 pi)^d f_Lambda(k^2)/(k^2+m^2)^(alpha+1)."""
    d = int(d)
    m = float(m)
    alpha = float(alpha)
    Lambda = float(Lambda)
    if regulator == "hard":
        prefactor = (
            2.0 * np.pi ** (d / 2.0)
            * Lambda ** d
            / (d * (2.0 * np.pi) ** d * special.gamma(d / 2.0))
            / m ** (2.0 * (alpha + 1.0))
        )
        return prefactor * special.hyp2f1(
            1.0 + alpha, d / 2.0, d / 2.0 + 1.0, -(Lambda ** 2) / (m ** 2)
        )
    if regulator == "gaussian":
        prefactor = (
            Lambda ** d
            / ((2.0 * np.pi) ** (d / 2.0) * (2.0 * Lambda * Lambda) ** (alpha + 1.0))
        )
        return prefactor * special.hyperu(
            1.0 + alpha, 2.0 + alpha - d / 2.0, m * m / (2.0 * Lambda * Lambda)
        )
    raise ValueError(f"regulator must be one of {_REGULATORS}, got {regulator!r}")


def propagator(r, d, m, Lambda=None, regulator=None):
    """Free scalar propagator
    G(r) = int d^d k/(2 pi)^d  f_Lambda(k^2) / (k^2 + m^2) e^{i k . r}.
    
    For regulator='none', this reduces to the standard free propagator:
        G(r) = (m / (2 pi r))^{(d/2)-1} K_{(d/2)-1}(m r) / (2 pi).
    For the other regulators, angular integration gives first
        int d Omega e^{i k . r} = (2 pi)^{d/2} (k r)^{1-(d/2)} J_{(d/2)-1}(k r),
    and the remaining radial integral is done numerically.
    """
    nu = d / 2.0 - 1.0
    if regulator == "none":
        if r < 1e-14:
            # raise ValueError("G2 diverges at r=0 for regulator='none'")
            return np.inf
        prefactor = (m / (2.0 * np.pi * r)) ** nu / (2.0 * np.pi)
        return prefactor * special.kv(nu, m * r)

    if regulator == "hard":
        k_max = float(Lambda)
    elif regulator == "gaussian":
        k_max = float(Lambda) * np.sqrt(2.0 * np.log(1.0 / 1e-14))
    else:
        raise ValueError(f"regulator must be one of {_REGULATORS}, got {regulator!r}")

    if r < 1e-14:
        return omega_alpha(d, m, alpha=0.0, Lambda=Lambda, regulator=regulator)
    else:
        prefactor = (2.0 * np.pi) ** (-d / 2.0) * r ** (1.0 - d / 2.0)

        def integrand(k):
            return (
                k ** (d / 2.0)
                * f_Lambda(k * k, Lambda, regulator)
                / (k * k + m * m)
                * special.jv(nu, k * r)
            )

        val, _ = integrate.quad(integrand, 0.0, k_max, limit=400)
        return prefactor * val


def G2_free(x1, x2, d, m, Lambda, regulator):
    """Free-theory 2-point function G^(2)(x1, x2).

    G(x1, x2) = int d^d k/(2 pi)^d  f_Lambda(k^2) / (k^2 + m^2) e^{i k . (x1 - x2)}.

    Each of x1, x2 may be a scalar (d=1) or a sequence of length d.
    """
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    x2 = np.asarray(x2, dtype=float).reshape(-1)
    r = float(np.linalg.norm(x1 - x2))
    return propagator(r, d, m, Lambda, regulator)


def G4_free(x1, x2, x3, x4, d, m, Lambda, regulator):
    """Free-theory 4-point function via Wick contractions.

    G^(4)(x1,x2,x3,x4) = G(x1,x2)G(x3,x4) + G(x1,x3)G(x2,x4) + G(x1,x4)G(x2,x3).

    Each of x1..x4 may be a scalar (d=1) or a sequence of length d.
    """
    pts = [np.asarray(x, dtype=float).reshape(-1) for x in (x1, x2, x3, x4)]
    rs = {
        (i, j): float(np.linalg.norm(pts[i] - pts[j]))
        for i in range(4) for j in range(i + 1, 4)
    }
    G = {ij: propagator(r, d, m, Lambda, regulator) for ij, r in rs.items()}
    return G[0, 1] * G[2, 3] + G[0, 2] * G[1, 3] + G[0, 3] * G[1, 2]
