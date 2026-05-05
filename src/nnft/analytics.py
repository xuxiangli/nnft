"""Analytic helpers for the field-theory NN architectures.

Houses closed-form / special-function expressions that are independent of any
particular sampler or architecture: UV regulators, the normalization constant
Omega_alpha, the target free-FT 2-point kernel, and the perturbative one-loop
expressions for lambda phi^4.
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


_GAUSS_TAIL_EPS = 1e-14


def _radial_kmax(Lambda, regulator):
    if regulator == "hard":
        return float(Lambda)
    if regulator == "gaussian":
        return float(Lambda) * np.sqrt(2.0 * np.log(1.0 / _GAUSS_TAIL_EPS))
    raise ValueError(f"regulator must be one of {_REGULATORS}, got {regulator!r}")


def _radial_propagator(r, d, m, Lambda, regulator, power=1):
    """Radial kernel
        I_p(r) = int d^d k/(2 pi)^d  f_Lambda(k^2) / (k^2 + m^2)^p e^{i k . r}
              = r^(1-d/2) (2 pi)^(-d/2) int_0^inf dk k^(d/2) f_Lambda(k^2) /
                                                  (k^2+m^2)^p J_{d/2-1}(k r)
    valid for r > 0. r=0 returns omega_alpha(alpha = power-1).
    """
    nu = d / 2.0 - 1.0
    if regulator == "none" and power == 1:
        # if power != 1:
        #     raise NotImplementedError(
        #         "no-UV-regulator radial kernel only implemented for power=1"
        #     )
        if r < 1e-14:
            return np.inf
        prefactor = (m / (2.0 * np.pi * r)) ** nu / (2.0 * np.pi)
        return prefactor * special.kv(nu, m * r)

    if r < 1e-14:
        return omega_alpha(d, m, alpha=power - 1.0, Lambda=Lambda, regulator=regulator)

    k_max = _radial_kmax(Lambda, regulator)
    prefactor = (2.0 * np.pi) ** (-d / 2.0) * r ** (1.0 - d / 2.0)

    def integrand(k):
        return (
            k ** (d / 2.0)
            * f_Lambda(k * k, Lambda, regulator)
            / (k * k + m * m) ** power
            * special.jv(nu, k * r)
        )

    val, _ = integrate.quad(integrand, 0.0, k_max, limit=400)
    return prefactor * val


def propagator(r, d, m, Lambda=None, regulator=None):
    """Free scalar propagator G(r) = int d^d k/(2 pi)^d f_Lambda(k^2)/(k^2+m^2) e^{ikr}.

    For regulator='none' this is (m/(2 pi r))^{d/2-1} K_{d/2-1}(m r) / (2 pi).
    """
    return _radial_propagator(r, d, m, Lambda, regulator, power=1)


def G2_free(x1, x2, d, m, Lambda, regulator):
    """Free-theory 2-point function G^(2)(x1, x2)."""
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    x2 = np.asarray(x2, dtype=float).reshape(-1)
    r = float(np.linalg.norm(x1 - x2))
    return propagator(r, d, m, Lambda, regulator)


def G4_free(x1, x2, x3, x4, d, m, Lambda, regulator):
    """Free-theory 4-point function via Wick contractions."""
    pts = [np.asarray(x, dtype=float).reshape(-1) for x in (x1, x2, x3, x4)]
    rs = {
        (i, j): float(np.linalg.norm(pts[i] - pts[j]))
        for i in range(4) for j in range(i + 1, 4)
    }
    G = {ij: propagator(r, d, m, Lambda, regulator) for ij, r in rs.items()}
    return G[0, 1] * G[2, 3] + G[0, 2] * G[1, 3] + G[0, 3] * G[1, 2]


def propagator_resummed(r, d, m, lambda_, Lambda, regulator):
    """Propagator with the one-loop tadpole resummed into a mass shift,
        m_eff^2 = m^2 + (lambda/2) * Delta(0).
    """
    Delta0 = propagator(0.0, d, m, Lambda, regulator)
    m_eff_sq = m * m + 0.5 * lambda_ * Delta0
    if m_eff_sq <= 0.0:
        raise ValueError(f"resummed mass^2 = {m_eff_sq} <= 0")
    return propagator(r, d, np.sqrt(m_eff_sq), Lambda, regulator)


def G2_lambda_phi4_one_loop_no_ir(x1, x2, d, m, lambda_, Lambda, regulator):
    """Tree + one-loop tadpole, no IR vertex regulator (translation invariant):
        G2(r) = Delta(r) - (lambda/2) Delta(0) I_2(r)
    where I_2(r) = int d^d k/(2 pi)^d f_Lambda(k^2) e^{ikr} / (k^2+m^2)^2.
    """
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    x2 = np.asarray(x2, dtype=float).reshape(-1)
    r = float(np.linalg.norm(x1 - x2))
    tree = _radial_propagator(r, d, m, Lambda, regulator, power=1)
    Delta0 = _radial_propagator(0.0, d, m, Lambda, regulator, power=1)
    I2 = _radial_propagator(r, d, m, Lambda, regulator, power=2)
    return tree - 0.5 * lambda_ * Delta0 * I2


def _hermite_nodes(n, d, L):
    """Tensor-product Gauss-Hermite nodes/weights for the measure
        int d^d y exp(-y^2/(2 L^2)) g(y) ~= sum_i W_i g(Y_i)
    with the normalization that sum_i W_i = (2 pi)^(d/2) L^d.
    """
    z, w = special.roots_hermite(int(n))
    # 1D: int dy exp(-y^2/2L^2) g(y) = L sqrt(2) sum_i w_i g(L sqrt(2) z_i).
    y1 = L * np.sqrt(2.0) * z
    w1 = L * np.sqrt(2.0) * w
    if d == 1:
        return y1.reshape(-1, 1), w1
    grids = np.meshgrid(*([y1] * d), indexing="ij")
    nodes = np.stack([g.reshape(-1) for g in grids], axis=-1)  # (n^d, d)
    w_grids = np.meshgrid(*([w1] * d), indexing="ij")
    weights = np.ones(n ** d)
    for wg in w_grids:
        weights *= wg.reshape(-1)
    return nodes, weights


def G2_lambda_phi4_one_loop_ir(
    x1, x2, d, m, lambda_, L, Lambda, regulator, n_hermite=24
):
    """Tree + one-loop tadpole with Gaussian IR vertex regulator:
        G2(x1, x2) = Delta(x1-x2) - (lambda/2) Delta(0) *
                     int d^d y exp(-y^2/2L^2) Delta(x1-y) Delta(y-x2).
    The y integral is by tensor-product Gauss-Hermite quadrature.
    Translation invariance is broken by the IR regulator.
    """
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    x2 = np.asarray(x2, dtype=float).reshape(-1)
    r = float(np.linalg.norm(x1 - x2))
    tree = _radial_propagator(r, d, m, Lambda, regulator, power=1)
    Delta0 = _radial_propagator(0.0, d, m, Lambda, regulator, power=1)

    nodes, weights = _hermite_nodes(n_hermite, d, L)
    acc = 0.0
    for y, w in zip(nodes, weights):
        r1 = float(np.linalg.norm(x1 - y))
        r2 = float(np.linalg.norm(y - x2))
        acc += w * propagator(r1, d, m, Lambda, regulator) * propagator(
            r2, d, m, Lambda, regulator
        )
    return tree - 0.5 * lambda_ * Delta0 * acc
