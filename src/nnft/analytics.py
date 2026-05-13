"""Analytic correlators for the field-theory NN architectures.

Two classes group the analytics by theory:

- :class:`AnalFree`  — free scalar 2- and 4-point.
- :class:`AnalPhi4`  — :math:`\\lambda \\phi^4` 2-point at one loop, with and
  without an IR vertex regulator.

Each correlator method exposes a ``method=`` argument selecting the
calculation route (Bessel-K closed form, radial-J numerical integral,
asymptotic expansions, Schwinger parameterization, Hermite quadrature,
Monte Carlo). Cross-method agreement is the verification target — see
``task/20260507_analytic/``.

Module-level helpers ``f_Lambda``, ``omega_alpha``, and the wrappers
``propagator``, ``G2_free``, ``G4_free``, ``G2_lambda_phi4_one_loop_no_ir``,
``G2_lambda_phi4_one_loop_ir``, ``propagator_resummed`` are preserved for
backward compatibility with ``architectures.py`` and existing scripts.

This module is intentionally NumPy/SciPy-based: most special functions
used here (``hyp2f1``, ``hyperu``, ``exp1``, ``gammaincc``, ``kv``, ``jv``)
and adaptive quadrature (``integrate.quad``/``dblquad``) have no JAX
equivalents. Inputs are accepted as ``np`` or ``jnp`` arrays and outputs
are ``np`` / Python floats; cast at call sites if a ``jnp`` value is
needed.
"""

import numpy as np
from scipy import integrate, special

_REGULATORS = ("hard", "gaussian", "none")


# =============================================================================
# Module-level helpers (used by both classes)
# =============================================================================


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


def _upper_gamma(s, x):
    """Upper incomplete gamma Γ(s, x) = ∫_x^∞ t^{s-1} e^{-t} dt for x > 0.

    Handles negative non-integer s via the recursion
        Γ(s, x) = (Γ(s+1, x) - x^s e^{-x}) / s ,
    and s == 0 via E_1(x) = scipy.special.exp1(x). Integer negative s is
    not used in this module.
    """
    if abs(s) < 1e-12:
        return float(special.exp1(x))
    if s > 0:
        return float(special.gamma(s) * special.gammaincc(s, x))
    return (_upper_gamma(s + 1.0, x) - x ** s * np.exp(-x)) / s


def _bessel_K_propagator(r, d, m):
    """Free propagator without UV regulator: closed-form Bessel-K."""
    nu = d / 2.0 - 1.0
    if r < 1e-14:
        return np.inf
    return (m / (2.0 * np.pi * r)) ** nu * special.kv(nu, m * r) / (2.0 * np.pi)


def _radial_J_integral(r, d, m, Lambda, regulator, power=1):
    """Generic radial integral
        I_p(r) = r^{1-d/2} (2 pi)^{-d/2} int_0^inf dk k^{d/2}
                                          f_Lambda(k^2) (k^2+m^2)^{-p} J_{d/2-1}(k r)
    """
    nu = d / 2.0 - 1.0
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


# =============================================================================
# AnalFree — free theory
# =============================================================================


class AnalFree:
    """Free scalar two-point and four-point analytics.

    Parameters
    ----------
    d : int
        Spatial dimension.
    m : float
        Mass.
    Lambda : float or None
        UV scale. Required unless ``regulator='none'``.
    regulator : {'gaussian', 'hard', 'none'}
        UV regulator. ``'none'`` admits only the closed-form ``bessel_K`` method.
    """

    def __init__(self, d, m, Lambda=None, regulator="gaussian"):
        if regulator not in _REGULATORS:
            raise ValueError(f"regulator must be one of {_REGULATORS}, got {regulator!r}")
        if regulator != "none" and Lambda is None:
            raise ValueError("Lambda must be supplied for a non-trivial regulator")
        self.d = int(d)
        self.m = float(m)
        self.Lambda = None if Lambda is None else float(Lambda)
        self.regulator = regulator
        self.nu = self.d / 2.0 - 1.0
        self._g2_table = None  # (log_r_grid, G_grid, r_min, r_max)

    def _g2_lookup(self, rs, r_max_hint=10.0, n_grid=400):
        """Vectorized G2 over an array of radii via a cached log-r table."""
        rs = np.asarray(rs, dtype=float)
        rmax = max(float(rs.max()) * 1.05, r_max_hint)
        if (self._g2_table is None) or (rmax > self._g2_table[3]):
            r_grid = np.geomspace(1e-6, rmax, n_grid)
            G_grid = np.array([self.G2(float(r)) for r in r_grid])
            self._g2_table = (np.log(r_grid), G_grid, float(r_grid[0]), float(r_grid[-1]))
        log_r, G, rmin, rmx = self._g2_table
        rs_c = np.clip(rs, rmin, rmx)
        return np.interp(np.log(rs_c), log_r, G)

    # ---- 2-point ----------------------------------------------------------

    def G2(self, r, method="auto"):
        """Free 2-point function G^(2)(|x1-x2|) = I_1(r).

        method
            ``'bessel_K'`` — closed form (regulator='none' only)
            ``'radial_J'`` — numerical 1D radial integral
            ``'asymptotic_large_r'`` — leading r >> 1/Lambda Gaussian-UV form
            ``'asymptotic_small_r'`` — leading r << 1/Lambda Gaussian-UV form
            ``'auto'`` — bessel_K when regulator='none', else asymptotic_large_r
        """
        r = float(r)
        if method == "auto":
            if self.regulator == "none":
                method = "bessel_K"
            else:
                method = "asymptotic_large_r" if r > 5 / self.Lambda else "radial_J"
        if method == "bessel_K":
            if self.regulator != "none":
                raise ValueError("'bessel_K' requires regulator='none'")
            return _bessel_K_propagator(r, self.d, self.m)
        if method == "radial_J":
            if self.regulator == "none":
                if r < 1e-14:
                    return np.inf
                return _bessel_K_propagator(r, self.d, self.m)
            return _radial_J_integral(r, self.d, self.m, self.Lambda, self.regulator, power=1)
        if method == "asymptotic_large_r":
            return self._G2_asymptotic_large_r(r)
        if method == "asymptotic_small_r":
            return self._G2_asymptotic_small_r(r)
        raise ValueError(f"unknown method {method!r}")

    def Delta0(self, method="auto"):
        if method == "auto":
            method = "bessel_K" if self.regulator == "none" else "radial_J"
        if self.regulator == "none":
            return np.inf
        return omega_alpha(self.d, self.m, alpha=0.0, Lambda=self.Lambda, regulator=self.regulator)

    # ---- 4-point ----------------------------------------------------------

    def G4(self, x1, x2, x3, x4, method="auto"):
        """Free 4-point via Wick contractions."""
        pts = [np.asarray(x, dtype=float).reshape(-1) for x in (x1, x2, x3, x4)]
        rs = {(i, j): float(np.linalg.norm(pts[i] - pts[j]))
              for i in range(4) for j in range(i + 1, 4)}
        G = {ij: self.G2(r, method=method) for ij, r in rs.items()}
        return G[0, 1] * G[2, 3] + G[0, 2] * G[1, 3] + G[0, 3] * G[1, 2]

    # ---- asymptotic forms -------------------------------------------------

    def _G2_asymptotic_large_r(self, r):
        """Leading large-r (r >> 1/Lambda) form for the Gaussian UV regulator,
            I_1(r) ≈ (2π)^{-d/2} (m/r)^ν K_ν(m r) e^{m²/(2Λ²)}.
        Equivalently: no-UV propagator multiplied by exp(m²/2Λ²).
        """
        if self.regulator != "gaussian":
            raise ValueError("'asymptotic_large_r' only defined for regulator='gaussian'")
        if r < 1e-14:
            return np.inf
        return (
            _bessel_K_propagator(r, self.d, self.m)
            * np.exp(self.m ** 2 / (2.0 * self.Lambda ** 2))
        )

    def _G2_asymptotic_small_r(self, r):
        """Leading small-r (r << 1/Lambda) form for the Gaussian UV regulator,

            I_1(r → 0) ≈ (4π)^{-d/2} m^{d-2} e^{m²/(2Λ²)}
                         · Γ(1 - d/2,  m²/(2Λ²)),

        derived by replacing J_ν(k r) ≈ (k r / 2)^ν / Γ(ν+1) (valid for k r ≪ 1
        on the support of f_Λ when r ≪ 1/Λ).  Independent of r at this order;
        sub-leading corrections are O((m r)², (m/Λ)²).
        """
        if self.regulator != "gaussian":
            raise ValueError("'asymptotic_small_r' only defined for regulator='gaussian'")
        x = self.m ** 2 / (2.0 * self.Lambda ** 2)
        y = (self.m * r / 2.0) ** 2
        s = 1.0 - self.d / 2.0
        gamma_inc = (
            _upper_gamma(s, x)
            - y * _upper_gamma(s - 1.0, x)
            + (y * y / 2.0) * _upper_gamma(s - 2.0, x)
            - (y * y * y / 6.0) * _upper_gamma(s - 3.0, x)
        )
        return (
            (4.0 * np.pi) ** (-self.d / 2.0)
            * self.m ** (self.d - 2.0)
            * np.exp(x)
            * gamma_inc
        )


# =============================================================================
# AnalPhi4 — λφ⁴ at one loop
# =============================================================================


class AnalPhi4:
    """λφ⁴ one-loop tadpole analytics.

    Parameters
    ----------
    free : AnalFree
        Underlying free theory; supplies d, m, Lambda, regulator.
    lambda_ : float
        Quartic coupling.
    L : float or None
        IR vertex regulator scale. ``L=None`` ⇒ no IR regulator
        (translation invariant).
    """

    def __init__(self, free, lambda_, L=None):
        self.free = free
        self.lambda_ = float(lambda_)
        self.L = None if L is None else float(L)
        self.d = free.d
        self.m = free.m
        self.Lambda = free.Lambda
        self.regulator = free.regulator

    # ---- tree -------------------------------------------------------------

    def G2_tree(self, r, method="auto"):
        return self.free.G2(r, method=method)

    # ---- one loop ---------------------------------------------------------

    def G2_one_loop(self, x1, x2, method="auto", **kw):
        """Tree + one-loop tadpole, i.e.
            G2(x1, x2) = Δ(|x1-x2|) - (λ/2) Δ(0) · I_loop(x1, x2; L).

        Without IR regulator (``L=None``):
            method ∈ {'radial_J', 'schwinger_t', 'deriv_alpha1'}, all
            translation invariant — depend on |x1 - x2|.
            ``radial_J`` and ``schwinger_t`` use the rescaled Λ → Λ/√2 because
            the two propagator regulators combine as e^{-k²/Λ²}.

        With IR regulator (``L`` set):
            method ∈ {'hermite', 'mc_gaussian', 'schwinger_t1t2'}.
        """
        x1 = np.asarray(x1, dtype=float).reshape(-1)
        x2 = np.asarray(x2, dtype=float).reshape(-1)
        tree_r = float(np.linalg.norm(x1 - x2))
        tree = self.free.G2(tree_r)
        Delta0 = self.free.Delta0()
        if self.L is None:
            return tree - 0.5 * self.lambda_ * Delta0 * self._loop_no_ir(tree_r, method, **kw)
        return tree - 0.5 * self.lambda_ * Delta0 * self._loop_ir(x1, x2, method, **kw)

    # ---- no-IR loop methods ----------------------------------------------

    def _loop_no_ir(self, r, method, **kw):
        if method == "auto":
            method = "schwinger_t"
        if method == "radial_J":
            return self._I2_radial_J(r)
        if method == "schwinger_t":
            return self._I2_schwinger_t(r)
        if method == "deriv_alpha1":
            return self._I2_deriv_alpha1(r, **kw)
        raise ValueError(f"unknown no-IR method {method!r}")

    def _Lambda_loop(self):
        """Effective Λ in I_2(r): two propagator regulators e^{-k²/2Λ²} combine
        to e^{-k²/Λ²} = e^{-k²/(2 (Λ/√2)²)}. So the loop integrand sees Λ/√2.
        """
        if self.regulator != "gaussian":
            return self.Lambda
        return self.Lambda / np.sqrt(2.0)

    def _I2_radial_J(self, r):
        Lambda_eff = self._Lambda_loop()
        if r < 1e-14:
            return omega_alpha(self.d, self.m, alpha=1.0, Lambda=Lambda_eff,
                               regulator=self.regulator)
        return _radial_J_integral(r, self.d, self.m, Lambda_eff, self.regulator, power=2)

    def _I2_schwinger_t(self, r, alpha=2):
        """Schwinger parameterization (Gaussian UV regulator only):
            I_α(r) = (2π)^{-d/2} / (2^α Γ(α))
                     · ∫_0^∞ dt t^{α-1} (t + 1/Λ²)^{-d/2}
                       · exp(-m² t / 2 - r² / (2(t + 1/Λ²))) ,
        with the loop-effective Λ_eff = Λ/√2 when called from the one-loop
        integrand.
        """
        if self.regulator != "gaussian":
            raise NotImplementedError("schwinger_t requires regulator='gaussian'")
        Lambda_eff = self._Lambda_loop()
        inv_L2 = 1.0 / (Lambda_eff ** 2)
        d = self.d
        m = self.m
        prefactor = (2.0 * np.pi) ** (-d / 2.0) / (2.0 ** alpha * special.gamma(alpha))

        def integrand(t):
            tp = t + inv_L2
            return (
                t ** (alpha - 1.0)
                * tp ** (-d / 2.0)
                * np.exp(-0.5 * m * m * t - 0.5 * r * r / tp)
            )

        val, _ = integrate.quad(integrand, 0.0, np.inf, limit=400)
        return prefactor * val

    def _I2_deriv_alpha1(self, r, h=None):
        """I_2(r) = -(1/(2 m)) ∂_m I_1(r), evaluated by central FD on the
        radial-J form of I_1 with Λ_eff = Λ/√2 (same Λ as the I_2 integrand).
        """
        Lambda_eff = self._Lambda_loop()
        m = self.m
        if h is None:
            h = 1e-3 * m
        if self.regulator == "none":
            def I1(mm):
                return _bessel_K_propagator(r, self.d, mm)
        else:
            def I1(mm):
                return _radial_J_integral(r, self.d, mm, Lambda_eff, self.regulator, power=1)
        return -(I1(m + h) - I1(m - h)) / (4.0 * m * h)

    # ---- IR loop methods --------------------------------------------------

    def _loop_ir(self, x1, x2, method, **kw):
        if method == "auto":
            method = "schwinger_t1t2"
        if method == "hermite":
            return self._loop_ir_hermite(x1, x2, **kw)
        if method == "mc_gaussian":
            return self._loop_ir_mc(x1, x2, **kw)
        if method == "schwinger_t1t2":
            return self._loop_ir_schwinger_t1t2(x1, x2, **kw)
        raise ValueError(f"unknown IR method {method!r}")

    def _loop_ir_hermite(self, x1, x2, n_hermite=48, sigma=None):
        """Coordinate-space integral
            I = ∫ d^d y exp(-y²/2L²) Δ(x1-y) Δ(y-x2),
        evaluated via a translated Hermite quadrature with reweighting.
        """
        d, m, L = self.d, self.m, self.L
        r = float(np.linalg.norm(x1 - x2))
        if sigma is None:
            sigma = max(r / 2.0, 1.0 / (2.0 * m))
        sigma = min(float(sigma), float(L))
        ymid = 0.5 * (x1 + x2)

        z, w = special.roots_hermite(int(n_hermite))
        if d == 1:
            nodes = (ymid[0] + sigma * np.sqrt(2.0) * z).reshape(-1, 1)
            ref_weights = sigma * np.sqrt(2.0) * w
        else:
            z_grid = np.meshgrid(*([z] * d), indexing="ij")
            w_grid = np.meshgrid(*([w] * d), indexing="ij")
            nodes = np.stack(
                [ymid[k] + sigma * np.sqrt(2.0) * zg.reshape(-1)
                 for k, zg in enumerate(z_grid)],
                axis=-1,
            )
            ref_weights = (sigma * np.sqrt(2.0)) ** d * np.ones(n_hermite ** d)
            for wg in w_grid:
                ref_weights = ref_weights * wg.reshape(-1)

        diff = nodes - ymid
        expo = (diff * diff).sum(axis=-1) / (2.0 * sigma * sigma) - (
            nodes * nodes
        ).sum(axis=-1) / (2.0 * L * L)

        # Vectorize the per-node propagator lookup via the cached G2 table.
        r1s = np.linalg.norm(x1[None, :] - nodes, axis=-1)
        r2s = np.linalg.norm(nodes - x2[None, :], axis=-1)
        r_hint = max(float(r1s.max()), float(r2s.max())) * 1.05 + 1e-3
        g1 = self.free._g2_lookup(r1s, r_max_hint=r_hint)
        g2 = self.free._g2_lookup(r2s, r_max_hint=r_hint)
        return float(np.sum(ref_weights * np.exp(expo) * g1 * g2))

    def _loop_ir_mc(self, x1, x2, M=int(2e5), seed=0, n_grid=400, return_err=False,
                    proposal="gaussian"):
        """Monte-Carlo estimate of
            I = ∫ d^d y exp(-y²/2L²) Δ(x1-y) Δ(y-x2) .

        Two proposal distributions are supported via ``proposal``:

        - ``"gaussian"`` — sample y ~ N(0, L² I_d).  This is the *literal*
          IR Gaussian, so the integrand becomes (2π L²)^{d/2} Δ Δ.
          Variance is enormous in regimes where the propagator support
          (scale ~ 1/m around the midpoint) does not overlap the IR
          support (scale L); use ``"matched"`` there.
        - ``"matched"`` — importance sample y ~ N(ymid, σ² I_d) with
          σ = min(max(r/2, 1/(2m)), L), then reweight by the IR Gaussian.
          This is the random-sample analogue of the Hermite scheme and
          gives controlled variance for any (r, L).
        """
        d, L = self.d, self.L
        x1 = np.asarray(x1, dtype=float).reshape(-1)
        x2 = np.asarray(x2, dtype=float).reshape(-1)
        rng = np.random.default_rng(seed)
        if proposal == "gaussian":
            ys = rng.normal(0.0, L, size=(M, d))
            log_w = np.zeros(M)
            prefactor = (2.0 * np.pi * L * L) ** (d / 2.0)
        elif proposal == "matched":
            r = float(np.linalg.norm(x1 - x2))
            sigma = min(max(r / 2.0, 1.0 / (2.0 * self.m)), float(L))
            ymid = 0.5 * (x1 + x2)
            ys = ymid[None, :] + sigma * rng.standard_normal((M, d))
            # importance weight: e^{-y²/(2L²)} / [N(y; ymid, σ²) / (2πσ²)^{d/2}]
            # writing the integrand as (2πσ²)^{d/2} N(y; ymid, σ²) * w(y) * Δ Δ
            # with w(y) = exp((y-ymid)²/(2σ²) - y²/(2L²))
            diff = ys - ymid[None, :]
            log_w = (diff * diff).sum(axis=-1) / (2.0 * sigma * sigma) \
                    - (ys * ys).sum(axis=-1) / (2.0 * L * L)
            prefactor = (2.0 * np.pi * sigma * sigma) ** (d / 2.0)
        else:
            raise ValueError(f"unknown proposal {proposal!r}")
        r1s = np.linalg.norm(x1[None, :] - ys, axis=1)
        r2s = np.linalg.norm(ys - x2[None, :], axis=1)
        r_hint = max(float(np.max(r1s)), float(np.max(r2s))) * 1.05 + 1e-3
        g1 = self.free._g2_lookup(r1s, r_max_hint=r_hint, n_grid=n_grid)
        g2 = self.free._g2_lookup(r2s, r_max_hint=r_hint, n_grid=n_grid)
        vals = np.exp(log_w) * g1 * g2
        mean = prefactor * vals.mean()
        if return_err:
            err = prefactor * vals.std(ddof=1) / np.sqrt(M)
            return mean, err
        return mean

    def _loop_ir_schwinger_t1t2(self, x1, x2, epsabs=1e-9, epsrel=1e-7):
        """Direct Schwinger integral over (t1, t2) for the Gaussian-UV /
        Gaussian-IR loop integral. Per ``notes/markdown/analytics.md``,

            -λ/2 Δ(0) (L²/2π)^{d/2} ∫₀^∞ dt₁ dt₂ D^{-d/2} exp[
                  -m²(t₁+t₂)
                  - ((B+C)x1² + (A+C)x2² - 2 C x1·x2) / (2D) ],

            A = 2 t₁ + 1/Λ², B = 2 t₂ + 1/Λ², C = L², D = AB + C(A+B).

        We return the bare double integral (without the -λ/2 Δ(0) prefactor),
        rescaled by (L²/2π)^{d/2}, so the caller's
            "-(λ/2) Δ(0) · _loop_ir(...)"
        gives the full one-loop correction.
        """
        if self.regulator != "gaussian":
            raise NotImplementedError("schwinger_t1t2 requires regulator='gaussian'")
        d, m, L, Lambda = self.d, self.m, self.L, self.Lambda
        inv_L2 = 1.0 / (Lambda * Lambda)
        Csq = L * L
        x1sq = float(np.dot(x1, x1))
        x2sq = float(np.dot(x2, x2))
        x12 = float(np.dot(x1, x2))

        def integrand_uv(s1, s2):
            # compactify [0, ∞) -> [0, 1) via t = s/(1-s)
            t1 = s1 / (1.0 - s1)
            t2 = s2 / (1.0 - s2)
            jac = 1.0 / ((1.0 - s1) ** 2 * (1.0 - s2) ** 2)
            A = 2.0 * t1 + inv_L2
            B = 2.0 * t2 + inv_L2
            C = Csq
            D = A * B + C * (A + B)
            num = (B + C) * x1sq + (A + C) * x2sq - 2.0 * C * x12
            return jac * D ** (-d / 2.0) * np.exp(-m * m * (t1 + t2) - 0.5 * num / D)

        val, _ = integrate.dblquad(
            integrand_uv, 0.0, 1.0, lambda s: 0.0, lambda s: 1.0,
            epsabs=epsabs, epsrel=epsrel,
        )
        prefactor = (Csq / (2.0 * np.pi)) ** (d / 2.0)
        return prefactor * val


# =============================================================================
# Backward-compatible function-style API
# =============================================================================


def propagator(r, d, m, Lambda=None, regulator=None):
    """Free scalar propagator G(r) = int d^d k/(2 pi)^d f_Lambda(k^2)/(k^2+m^2) e^{ikr}.

    For regulator='none' this is (m/(2 pi r))^{d/2-1} K_{d/2-1}(m r) / (2 pi).
    """
    if regulator == "none":
        if r < 1e-14:
            return np.inf
        return _bessel_K_propagator(r, d, m)
    if r < 1e-14:
        return omega_alpha(d, m, alpha=0.0, Lambda=Lambda, regulator=regulator)
    return _radial_J_integral(r, d, m, Lambda, regulator, power=1)


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
    """Tree + one-loop tadpole, no IR vertex regulator (translation invariant).
    The loop-side regulator combines as Λ → Λ/√2 (Gaussian only).
    """
    free = AnalFree(d, m, Lambda, regulator)
    phi4 = AnalPhi4(free, lambda_, L=None)
    return phi4.G2_one_loop(x1, x2, method="radial_J")


def G2_lambda_phi4_one_loop_ir(
    x1, x2, d, m, lambda_, L, Lambda, regulator, n_hermite=48, sigma=None
):
    """Tree + one-loop tadpole with Gaussian IR vertex regulator
    (Hermite-quadrature evaluation of the y-integral).
    """
    free = AnalFree(d, m, Lambda, regulator)
    phi4 = AnalPhi4(free, lambda_, L=L)
    return phi4.G2_one_loop(x1, x2, method="hermite",
                            n_hermite=n_hermite, sigma=sigma)
