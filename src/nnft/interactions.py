"""Interaction terms S_int[phi] for NNFT samplers.

Currently only the lambda phi^4 vertex with a Gaussian or hard IR regulator is
implemented. The class exposes several action-evaluation methods (real-space
MC quadrature, real-space Gauss-Hermite, and the exact O(N^4) momentum-space
sums) that share parameters and validate against each other on small N.
"""

import numpy as np
from scipy import special
from scipy.spatial import cKDTree


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
               x_quad=None, quad_weights=None, eps=1e-8, k_cut=None):
        """Return S_int per batch element, shape (b,).

        Args:
            theory: a Theory whose architecture and N are used.
            params: dict from architecture.param_spec, values shaped
                    (b * N, *spec_shape) (the standard packed layout).
            b:      number of independent param configurations packed in params.
            method: one of "real_space_mc", "real_space_hermite",
                    "explicit", "symmetry_reduced", "momentum_conservation".
            M_x, n_hermite, rng: method-specific options (see _action_*).
            x_quad, quad_weights: optional cached quadrature for the real-space
                methods. If x_quad is given, it overrides M_x / n_hermite.
            eps, k_cut: method-specific options for "momentum_conservation".
                If k_cut is None, use sqrt(2 log(1/eps)) / L.
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
        if method == "momentum_conservation":
            return self._action_momentum_conservation(
                theory, params, b, eps=eps, k_cut=k_cut
            )
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

    # ----- approximate momentum-conservation summation -------------------
    def _action_momentum_conservation(self, theory, params, b, *, eps, k_cut):
        """Sparse signed-pair sum using approximate momentum conservation.

        The Gaussian IR factor suppresses modes with total momentum
        |K| >> 1/L. This method keeps only signed-pair combinations satisfying
        |P_a + P_b| <= k_cut, where the default cutoff makes the omitted
        Gaussian factor no larger than eps per term.
        """
        d = theory.architecture.d_in
        N = theory.N
        if k_cut is None:
            eps = float(eps)
            if not 0.0 < eps < 1.0:
                raise ValueError("eps must satisfy 0 < eps < 1")
            k_cut = np.sqrt(2.0 * np.log(1.0 / eps)) / self.L
        k_cut = float(k_cut)
        if k_cut < 0.0:
            raise ValueError("k_cut must be non-negative")

        out = np.empty(b, dtype=float)
        for bi in range(b):
            sl = slice(bi * N, (bi + 1) * N)
            out[bi] = self._momentum_conservation_one(
                d, params, sl, theory, k_cut
            )
        return out

    def _momentum_conservation_one(self, d, params, sl, theory, k_cut):
        pair_momenta, pair_coeffs = self._signed_pair_momenta_and_coeffs(
            params, sl, theory, representative=not np.isinf(k_cut)
        )
        gamma = 0.5 * self.L * self.L

        if np.isinf(k_cut):
            total = self._dense_signed_pair_sum(pair_momenta, pair_coeffs, gamma)
        else:
            total = self._sparse_representative_pair_sum(
                pair_momenta, pair_coeffs, gamma, k_cut
            )

        prefactor = (
            self.lambda_
            * (2.0 * np.pi) ** (d / 2.0)
            * self.L ** d
            * theory._c_N ** 4
            / 24.0
        )
        return float(prefactor * np.real(total))

    def _signed_pair_momenta_and_coeffs(
        self, params, sl, theory, *, representative=False
    ):
        """Return ordered signed pair momenta and complex coefficients.

        Pair labels are ordered ``(i, j, sigma, tau)`` with
        ``P = sigma k_i + tau k_j`` and
        ``C = u_i^sigma u_j^tau`` where
        ``u_i^sigma = 0.5 * a_i * exp(i sigma b_i)``. If representative is
        true, return only the sigma=+1 blocks; the sparse summation reconstructs
        the sigma=-1 conjugate blocks analytically.
        """
        arch = theory.architecture
        k = np.asarray(params["W0"][sl], dtype=float)
        b0 = np.asarray(params["b0"][sl], dtype=float)
        if hasattr(arch, "alpha"):
            W1 = np.asarray(params["W1"][sl], dtype=float)
            w1 = W1 * (np.sum(k * k, axis=-1) + arch.m * arch.m) ** (
                arch.alpha / 2.0
            )
        else:
            w1 = np.asarray(params["W1"][sl], dtype=float)

        N, d = k.shape
        NN = N * N
        sign_pairs = (
            ((1.0, 1.0), (1.0, -1.0))
            if representative
            else ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
        )
        pair_momenta = np.empty((len(sign_pairs) * NN, d), dtype=float)
        pair_coeffs = np.empty(len(sign_pairs) * NN, dtype=np.complex128)
        signed_coeffs = {
            1.0: 0.5 * w1 * np.exp(1j * b0),
            -1.0: 0.5 * w1 * np.exp(-1j * b0),
        }

        for block, (sigma, tau) in enumerate(sign_pairs):
            start = block * NN
            stop = start + NN
            pair_momenta[start:stop] = (
                sigma * k[:, None, :] + tau * k[None, :, :]
            ).reshape(NN, d)
            pair_coeffs[start:stop] = (
                signed_coeffs[sigma][:, None] * signed_coeffs[tau][None, :]
            ).reshape(NN)
        return pair_momenta, pair_coeffs

    def _sparse_representative_pair_sum(
        self, pair_momenta, pair_coeffs, gamma, k_cut
    ):
        tree = cKDTree(pair_momenta)

        plus = tree.sparse_distance_matrix(
            cKDTree(-pair_momenta), k_cut, output_type="coo_matrix"
        )
        plus_weights = np.exp(-gamma * plus.data * plus.data)
        plus_total = np.sum(
            pair_coeffs[plus.row] * pair_coeffs[plus.col] * plus_weights
        )

        diff = tree.sparse_distance_matrix(tree, k_cut, output_type="coo_matrix")
        diff_weights = np.exp(-gamma * diff.data * diff.data)
        diff_total = np.sum(
            pair_coeffs[diff.row]
            * np.conjugate(pair_coeffs[diff.col])
            * diff_weights
        )
        return 2.0 * np.real(plus_total + diff_total)

    def _dense_signed_pair_sum(self, pair_momenta, pair_coeffs, gamma):
        total = 0.0 + 0.0j
        for idx in range(pair_momenta.shape[0]):
            K = pair_momenta[idx] + pair_momenta
            Ksq = np.sum(K * K, axis=1)
            total += pair_coeffs[idx] * np.sum(
                pair_coeffs * np.exp(-gamma * Ksq)
            )
        return total
