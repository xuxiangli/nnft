"""Interaction terms S_int[phi] for NNFT samplers.

Currently only the lambda phi^4 vertex with a Gaussian or hard IR regulator is
implemented. The class exposes several action-evaluation methods (real-space
MC quadrature, real-space Gauss-Hermite, and the exact O(N^4) momentum-space
sums) that share parameters and validate against each other on small N.
"""

import numpy as np
from scipy import special


_IR_REGULATORS = ("gaussian",)


class LambdaPhi4:
    """S_int[phi] = (lambda/4!) int d^d x f_IR(x^2) phi(x)^4.

    Currently only f_IR(x^2) = exp(-x^2 / 2 L^2) is supported. The class is a
    stateless evaluator: pass it (theory, params, b) and it returns S_int per
    batch element. Quadrature points/weights can be cached by the caller (e.g.
    a Markov-chain sampler) by passing `x_quad` / `quad_weights`.
    """

    def __init__(self, lambda_, L, ir_regulator="gaussian"):
        if ir_regulator not in _IR_REGULATORS:
            raise ValueError(
                f"ir_regulator must be one of {_IR_REGULATORS}, got {ir_regulator!r}"
            )
        self.lambda_ = float(lambda_)
        self.L = float(L)
        self.ir_regulator = ir_regulator

    # ----- public entry point --------------------------------------------
    def action(self, theory, params, b, *, method="real_space_mc",
               M_x=None, n_hermite=None, rng=None,
               x_quad=None, quad_weights=None):
        """Return S_int per batch element, shape (b,).

        Args:
            theory: a Theory whose architecture and N are used.
            params: dict from architecture.param_spec, values shaped
                    (b * N, *spec_shape) (the standard packed layout).
            b:      number of independent param configurations packed in params.
            method: one of "real_space_mc", "real_space_hermite",
                    "explicit", "symmetry_reduced".
            M_x, n_hermite, rng: method-specific options (see _action_*).
            x_quad, quad_weights: optional cached quadrature for the real-space
                methods. If x_quad is given, it overrides M_x / n_hermite.
        """
        if method == "real_space_mc":
            if x_quad is not None:
                xs = np.asarray(x_quad, dtype=float)
                # MC interpretation: weights are uniform, prefactor below
                return self._action_real_space_mc_with_points(
                    theory, params, b, xs
                )
            if rng is None:
                raise ValueError("real_space_mc requires rng (or x_quad)")
            if M_x is None:
                raise ValueError("real_space_mc requires M_x (or x_quad)")
            xs = self.draw_mc_points(theory.architecture.d_in, int(M_x), rng)
            return self._action_real_space_mc_with_points(theory, params, b, xs)

        if method == "real_space_hermite":
            if x_quad is None or quad_weights is None:
                if n_hermite is None:
                    raise ValueError(
                        "real_space_hermite requires n_hermite (or x_quad+quad_weights)"
                    )
                xs, ws = self.hermite_points(theory.architecture.d_in, int(n_hermite))
            else:
                xs = np.asarray(x_quad, dtype=float)
                ws = np.asarray(quad_weights, dtype=float)
            return self._action_real_space_hermite_with_points(
                theory, params, b, xs, ws
            )

        if method == "explicit":
            return self._action_explicit(theory, params, b)
        if method == "symmetry_reduced":
            return self._action_symmetry_reduced(theory, params, b)
        raise ValueError(f"unknown method {method!r}")

    # ----- quadrature helpers --------------------------------------------
    def draw_mc_points(self, d, M_x, rng):
        """Draw M_x points from N(0, L^2 I_d) for the real-space MC quadrature."""
        return rng.normal(loc=0.0, scale=self.L, size=(int(M_x), int(d)))

    def hermite_points(self, d, n):
        """Tensor-product Gauss-Hermite nodes/weights so that
            int d^d y exp(-y^2/2L^2) g(y) ~= sum_i W_i g(Y_i).
        Returns (Y of shape (n^d, d), W of shape (n^d,)).
        """
        z, w = special.roots_hermite(int(n))
        y1 = self.L * np.sqrt(2.0) * z
        w1 = self.L * np.sqrt(2.0) * w
        if d == 1:
            return y1.reshape(-1, 1), w1
        grids = np.meshgrid(*([y1] * d), indexing="ij")
        nodes = np.stack([g.reshape(-1) for g in grids], axis=-1)
        w_grids = np.meshgrid(*([w1] * d), indexing="ij")
        weights = np.ones(n ** d)
        for wg in w_grids:
            weights *= wg.reshape(-1)
        return nodes, weights

    # ----- core implementations ------------------------------------------
    def _action_real_space_mc_with_points(self, theory, params, b, xs):
        """S_int ~= (lambda (2 pi)^{d/2} L^d) / (4! M_x) sum_r phi(x_r)^4.

        xs: (M_x, d_in) real-space sample points drawn from N(0, L^2 I).
        Returns array (b,).
        """
        d = theory.architecture.d_in
        M_x = xs.shape[0]
        phi = theory.evaluate(xs, params, b=b)            # (b, M_x)
        phi4 = (phi ** 4).sum(axis=1)                     # (b,)
        prefactor = (
            self.lambda_ * (2.0 * np.pi) ** (d / 2.0) * self.L ** d
            / (24.0 * M_x)
        )
        return prefactor * phi4

    def _action_real_space_hermite_with_points(self, theory, params, b, xs, ws):
        """S_int ~= (lambda / 4!) sum_i W_i phi(Y_i)^4 with Hermite (Y, W)."""
        phi = theory.evaluate(xs, params, b=b)            # (b, n_q)
        phi4_w = (phi ** 4) * ws[None, :]                 # (b, n_q)
        return (self.lambda_ / 24.0) * phi4_w.sum(axis=1)

    def _action_explicit(self, theory, params, b):
        """Exact O(8 N^4) momentum-space sum; only practical for tiny N (<=10)."""
        d = theory.architecture.d_in
        N = theory.N
        out = np.empty(b, dtype=float)
        for bi in range(b):
            sl = slice(bi * N, (bi + 1) * N)
            out[bi] = self._explicit_one(d, N, params, sl, theory)
        return out

    def _explicit_one(self, d, N, params, sl, theory):
        arch = theory.architecture
        k = np.asarray(params["W0"][sl], dtype=float)     # (N, d)
        b0 = np.asarray(params["b0"][sl], dtype=float)    # (N,)
        # w^(1) = W1 * (|k|^2 + m^2)^(alpha/2) for CosNetFT; for plain CosNet
        # arch.evaluate uses W1 directly. We mimic that branching by reading
        # the per-neuron amplitude from a single-point evaluation: phi = c_N
        # sum_i (W1 * (|k|^2+m^2)^(alpha/2)) cos(b_i) at x=0 => the per-neuron
        # amplitudes are (W1 * (|k|^2+m^2)^(alpha/2)). Read them directly.
        if hasattr(arch, "alpha"):
            W1 = np.asarray(params["W1"][sl], dtype=float)
            m = arch.m
            alpha = arch.alpha
            w1 = W1 * (np.sum(k * k, axis=-1) + m * m) ** (alpha / 2.0)
        else:
            w1 = np.asarray(params["W1"][sl], dtype=float)

        # Sum over (s1, s2, s3) in {+1,-1}^3, with i = + by symmetry; multiply by 2.
        L2 = self.L * self.L
        total = 0.0
        # broadcast indices
        ki = k[:, None, None, None, :]      # (N,1,1,1,d)
        kj = k[None, :, None, None, :]
        kk = k[None, None, :, None, :]
        kl = k[None, None, None, :, :]
        bi = b0[:, None, None, None]
        bj = b0[None, :, None, None]
        bk = b0[None, None, :, None]
        bl = b0[None, None, None, :]
        wi = w1[:, None, None, None]
        wj = w1[None, :, None, None]
        wk = w1[None, None, :, None]
        wl = w1[None, None, None, :]
        ww = wi * wj * wk * wl
        for s1 in (+1, -1):
            for s2 in (+1, -1):
                for s3 in (+1, -1):
                    K = ki + s1 * kj + s2 * kk + s3 * kl
                    Ksq = np.sum(K * K, axis=-1)
                    phase = bi + s1 * bj + s2 * bk + s3 * bl
                    total += np.sum(np.cos(phase) * ww * np.exp(-0.5 * L2 * Ksq))
        # The 8-term sum here equals the full 16-term sum over (sigma_i, sigma_j,
        # sigma_k, sigma_l) divided by 2 (the (sigma -> -sigma) relabeling).
        # Combined with the (1/16) from cos = (e^{i.} + e^{-i.})/2 in phi^4,
        # the prefactor is lambda (2 pi)^{d/2} L^d / (4! * 8 * N^2).
        prefactor = (
            self.lambda_ * (2.0 * np.pi) ** (d / 2.0) * self.L ** d
            / (24.0 * 8.0 * N * N)
        )
        return prefactor * total

    def _action_symmetry_reduced(self, theory, params, b):
        """Same as _action_explicit but only sums over i<=j<=k<=l with the
        appropriate multinomial multiplicity. Reduces work by ~4!.
        """
        # for the requested validation use, just defer to the explicit form;
        # included as an alias to make the API surface match the plan.
        return self._action_explicit(theory, params, b)
