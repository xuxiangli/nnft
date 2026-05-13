"""Interaction terms S_int[phi] for NNFT samplers (JAX backend).

Implements the lambda phi^4 vertex with a Gaussian IR regulator. Three
JAX-pure methods are exposed for use by samplers that need `jax.grad`:

    - ``real_space_mc``        : MC quadrature with key-drawn points.
    - ``real_space_hermite``   : tensor-product Gauss-Hermite quadrature.
    - ``trans_sym``            : dense signed-pair momentum-space sum,
                                  O(16 N^4) (no cKDTree).

Layout: each method takes a **single-configuration** params dict
{name -> (N, *spec)} and returns a scalar action. Batch evaluation is
done by the caller via `jax.vmap`. The NumPy/cKDTree sparse `trans_sym`
path from the main branch is not ported (JAX has no sparse-neighbour
primitive, and HMC/MALA need a differentiable evaluator).
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from scipy import special


_IR_REGULATORS = ("gaussian",)


class LambdaPhi4:
    """S_int[phi] = (lambda/4!) int d^d x exp(-x^2/(2 L^2)) phi(x)^4."""

    def __init__(self, lambda_, L, ir_regulator="gaussian"):
        if ir_regulator not in _IR_REGULATORS:
            raise ValueError(
                f"ir_regulator must be one of {_IR_REGULATORS}, "
                f"got {ir_regulator!r}"
            )
        self.lambda_ = float(lambda_)
        self.L = float(L)
        self.ir_regulator = ir_regulator

    # ----- public entry points -------------------------------------------
    def action(
        self,
        theory,
        params,
        *,
        method="trans_sym",
        M_x=None,
        n_hermite=None,
        key=None,
        x_quad=None,
        quad_weights=None,
        eps=1e-8,
        k_cut=None,
    ):
        """Return scalar S_int for a single parameter configuration.

        Args:
            theory:     Theory whose architecture and N are used.
            params:     dict; values shape (N, *spec_shape).
            method:     "real_space_mc" | "real_space_hermite" | "trans_sym".
            M_x:        MC quadrature size (real_space_mc).
            n_hermite:  Hermite nodes per dim (real_space_hermite).
            key:        jr.PRNGKey for real_space_mc point draws.
            x_quad, quad_weights: cached quadrature (overrides M_x/n_hermite).
            eps, k_cut: trans_sym truncation parameters (currently dense, so
                        these are accepted for API parity but unused).
        """
        del eps, k_cut  # dense trans_sym ignores truncation
        if method == "real_space_mc":
            if x_quad is not None:
                xs = jnp.asarray(x_quad)
            elif key is None or M_x is None:
                raise ValueError(
                    "real_space_mc requires key and M_x (or x_quad)"
                )
            else:
                xs = self.draw_mc_points(theory.architecture.d_in, int(M_x), key)
            return self._action_real_space_mc(theory, params, xs)

        if method == "real_space_hermite":
            if x_quad is None or quad_weights is None:
                if n_hermite is None:
                    raise ValueError(
                        "real_space_hermite requires n_hermite "
                        "(or x_quad+quad_weights)"
                    )
                xs, ws = self.hermite_points(
                    theory.architecture.d_in, int(n_hermite)
                )
            else:
                xs = jnp.asarray(x_quad)
                ws = jnp.asarray(quad_weights)
            return self._action_real_space_hermite(theory, params, xs, ws)

        if method == "trans_sym":
            return self._action_trans_sym_dense(theory, params)

        raise ValueError(f"unknown method {method!r}")

    def action_batched(self, theory, params, **kwargs):
        """Vectorised wrapper. `params` here is shape (b, N, *spec); returns (b,)."""
        per_one = lambda p: self.action(theory, p, **kwargs)
        return jax.vmap(per_one)(params)

    # ----- quadrature helpers --------------------------------------------
    def draw_mc_points(self, d, M_x, key):
        return self.L * jr.normal(key, (int(M_x), int(d)))

    def hermite_points(self, d, n):
        """Tensor-product Gauss-Hermite nodes/weights so that
            int d^d y exp(-y^2/(2 L^2)) g(y) ~= sum_i W_i g(Y_i).
        Returns (Y of shape (n^d, d), W of shape (n^d,)) as jnp arrays.
        """
        z, w = special.roots_hermite(int(n))
        y1 = self.L * np.sqrt(2.0) * z
        w1 = self.L * np.sqrt(2.0) * w
        if d == 1:
            return jnp.asarray(y1.reshape(-1, 1)), jnp.asarray(w1)
        grids = np.meshgrid(*([y1] * d), indexing="ij")
        nodes = np.stack([g.reshape(-1) for g in grids], axis=-1)
        w_grids = np.meshgrid(*([w1] * d), indexing="ij")
        weights = np.ones(n ** d)
        for wg in w_grids:
            weights *= wg.reshape(-1)
        return jnp.asarray(nodes), jnp.asarray(weights)

    # ----- core implementations (single-config) --------------------------
    def _action_real_space_mc(self, theory, params, xs):
        d = theory.architecture.d_in
        M_x = xs.shape[0]
        phi = theory.evaluate(xs, params)                  # (M_x,)
        prefactor = (
            self.lambda_ * (2.0 * np.pi) ** (d / 2.0) * self.L ** d
            / (24.0 * M_x)
        )
        return prefactor * jnp.sum(phi ** 4)

    def _action_real_space_hermite(self, theory, params, xs, ws):
        phi = theory.evaluate(xs, params)                  # (n_q,)
        return (self.lambda_ / 24.0) * jnp.sum(ws * phi ** 4)

    def _action_trans_sym_dense(self, theory, params):
        """Dense signed-pair momentum-space sum, O(16 N^4)."""
        d = theory.architecture.d_in
        gamma = 0.5 * self.L * self.L
        prefactor = (
            self.lambda_ * (2.0 * np.pi) ** (d / 2.0) * self.L ** d
            * theory._c_N ** 4
            / 24.0
        )
        P, C = self._signed_pair_momenta_and_coeffs(theory, params)
        P_sum = P[:, None, :] + P[None, :, :]
        K = jnp.exp(-gamma * jnp.sum(P_sum * P_sum, axis=-1))
        total = jnp.einsum("a,b,ab->", C, C, K)
        return prefactor * jnp.real(total)

    def _signed_pair_momenta_and_coeffs(self, theory, params):
        arch = theory.architecture
        k = params["W0"]                                    # (N, d)
        b0 = params["b0"]                                   # (N,)
        if hasattr(arch, "alpha"):
            scale = (jnp.sum(k * k, axis=-1) + arch.m * arch.m) ** (
                arch.alpha / 2.0
            )
            a = params["W1"] * scale
        else:
            a = params["W1"]

        u_plus = 0.5 * a * jnp.exp(1j * b0)
        u_minus = 0.5 * a * jnp.exp(-1j * b0)

        signs = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
        u_by_sign = {1.0: u_plus, -1.0: u_minus}
        d = k.shape[-1]
        blocks_P = []
        blocks_C = []
        for sigma, tau in signs:
            P = (sigma * k[:, None, :] + tau * k[None, :, :]).reshape(-1, d)
            C = (
                u_by_sign[sigma][:, None] * u_by_sign[tau][None, :]
            ).reshape(-1)
            blocks_P.append(P)
            blocks_C.append(C)
        return jnp.concatenate(blocks_P, axis=0), jnp.concatenate(
            blocks_C, axis=0
        )
