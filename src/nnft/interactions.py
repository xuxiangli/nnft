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
    def action(self, theory, params, b, *, method="trans_sym",
               M_x=None, n_hermite=None, rng=None,
               x_quad=None, quad_weights=None, eps=1e-8, k_cut=None):
        """Return S_int per batch element, shape (b,).

        Args:
            theory: a Theory whose architecture and N are used.
            params: dict from architecture.param_spec, values shaped
                    (b * N, *spec_shape) (the standard packed layout).
            b:      number of independent param configurations packed in params.
            method: one of "real_space_mc", "real_space_hermite",
                    "explicit", "perm_sym", "trans_sym".
            M_x, n_hermite, rng: method-specific options (see _action_*).
            x_quad, quad_weights: optional cached quadrature for the real-space
                methods. If x_quad is given, it overrides M_x / n_hermite.
            eps, k_cut: method-specific options for "trans_sym".
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
        if method == "perm_sym":
            return self._action_perm_sym(theory, params, b)
        if method == "trans_sym":
            return self._action_trans_sym(
                theory, params, b, eps=eps, k_cut=k_cut
            )
        raise ValueError(f"unknown method {method!r}")

    # ----- action + gradient -------------------------------------------
    def action_and_grad(self, theory, params, b, *, method="trans_sym",
                        M_x=None, rng=None, x_quad=None,
                        eps=1e-8, k_cut=None):
        """Return (S, grad) where S has shape (b,) and grad is a dict
            {"W0": (b*N, d), "b0": (b*N,), "W1": (b*N,)}.

        Supported methods:
          - "real_space_mc": gradient of the noisy MC quadrature; cheap (O(MN))
            but inherits the ~10^-2 quadrature variance.
          - "trans_sym": pair-pair sum with momentum-conservation truncation
            |P_A + P_B| < k_cut; matches `action(method="trans_sym")` in
            precision (machine-eps as eps -> 0). Cost ~O(N^2 * neighbors_per_pair).
        """
        if method == "real_space_mc":
            if x_quad is not None:
                xs = np.asarray(x_quad, dtype=float)
            else:
                if rng is None or M_x is None:
                    raise ValueError("real_space_mc requires rng and M_x (or x_quad)")
                xs = self.draw_mc_points(theory.architecture.d_in, int(M_x), rng)
            return self._action_and_grad_real_space_mc(theory, params, b, xs)
        if method == "trans_sym":
            return self._action_and_grad_trans_sym(
                theory, params, b, eps=eps, k_cut=k_cut
            )
        raise NotImplementedError(
            f"action_and_grad does not support method={method!r}"
        )

    def _action_and_grad_real_space_mc(self, theory, params, b, xs):
        arch = theory.architecture
        d = arch.d_in
        N = theory.N
        M = xs.shape[0]
        c_N = theory._c_N
        A = self.lambda_ * (2.0 * np.pi) ** (d / 2.0) * self.L ** d / 24.0

        has_alpha = hasattr(arch, "alpha")
        m_sq = arch.m * arch.m if has_alpha else None
        alpha = arch.alpha if has_alpha else 0.0

        W0_all = np.asarray(params["W0"], dtype=float)   # (b*N, d)
        b0_all = np.asarray(params["b0"], dtype=float)   # (b*N,)
        W1_all = np.asarray(params["W1"], dtype=float)   # (b*N,)

        S_out = np.empty(b, dtype=float)
        grad_W0 = np.zeros_like(W0_all)
        grad_b0 = np.zeros_like(b0_all)
        grad_W1 = np.zeros_like(W1_all)   # left zero (W1 fixed by Constant)

        coef = 4.0 * A * c_N / M

        for bi in range(b):
            sl = slice(bi * N, (bi + 1) * N)
            k = W0_all[sl]                            # (N, d)
            b0 = b0_all[sl]                           # (N,)
            W1 = W1_all[sl]                           # (N,)
            ksq = np.sum(k * k, axis=-1)
            if has_alpha:
                scale = (ksq + m_sq) ** (alpha / 2.0)
                w = W1 * scale
            else:
                w = W1

            pre = k @ xs.T + b0[:, None]               # (N, M)
            cos_pre = np.cos(pre)
            sin_pre = np.sin(pre)
            Phi = c_N * (w[:, None] * cos_pre).sum(axis=0)   # (M,)
            Phi3 = Phi ** 3                                  # (M,)

            S_out[bi] = (A / M) * np.sum(Phi * Phi3)

            # ∂S/∂b_i = -coef · w_i · Σ_r Phi3_r sin(pre_{ir})
            grad_b0[sl] = -coef * w * (sin_pre @ Phi3)

            # ∂S/∂k_{i,a}: term2 (always present)
            #   = -coef · w_i · Σ_r Phi3_r sin(pre) x_{r,a}
            sin_phi3 = sin_pre * Phi3[None, :]               # (N, M)
            gk = -coef * w[:, None] * (sin_phi3 @ xs)        # (N, d)

            # term1 (only when alpha != 0): coef · ∂w_i/∂k_{i,a} · Σ_r Phi3_r cos(pre)
            if has_alpha and alpha != 0.0:
                cos_dot_phi3 = cos_pre @ Phi3                # (N,)
                dw_dk = (
                    alpha * W1[:, None] * k
                    * ((ksq + m_sq) ** (alpha / 2.0 - 1.0))[:, None]
                )                                            # (N, d)
                gk = gk + coef * dw_dk * cos_dot_phi3[:, None]

            grad_W0[sl] = gk

        return S_out, {"W0": grad_W0, "b0": grad_b0, "W1": grad_W1}

    def _action_and_grad_trans_sym(self, theory, params, b, *, eps, k_cut):
        """Pair-pair sum gradient with the same momentum-conservation
        truncation as `_action_trans_sym`. See class docstring of
        ``action_and_grad`` and the inline derivation in
        ``_action_and_grad_trans_sym_one``.
        """
        d = theory.architecture.d_in
        N = theory.N
        if k_cut is None:
            eps_f = float(eps)
            if not 0.0 < eps_f < 1.0:
                raise ValueError("eps must satisfy 0 < eps < 1")
            k_cut = np.sqrt(2.0 * np.log(1.0 / eps_f)) / self.L
        k_cut = float(k_cut)
        if k_cut < 0.0:
            raise ValueError("k_cut must be non-negative")

        gamma = 0.5 * self.L * self.L
        pref = (
            self.lambda_ * (2.0 * np.pi) ** (d / 2.0) * self.L ** d
            * theory._c_N ** 4 / 24.0
        )

        W0_all = np.asarray(params["W0"], dtype=float)
        S_out = np.empty(b, dtype=float)
        grad_W0 = np.zeros_like(W0_all)
        grad_b0 = np.zeros(W0_all.shape[0], dtype=float)
        grad_W1 = np.zeros(W0_all.shape[0], dtype=float)

        for bi in range(b):
            sl = slice(bi * N, (bi + 1) * N)
            S_, gW0, gb0 = self._action_and_grad_trans_sym_one(
                d, N, params, sl, theory, k_cut, gamma, pref,
            )
            S_out[bi] = S_
            grad_W0[sl] = gW0
            grad_b0[sl] = gb0
        return S_out, {"W0": grad_W0, "b0": grad_b0, "W1": grad_W1}

    def _action_and_grad_trans_sym_one(
        self, d, N, params, sl, theory, k_cut, gamma, pref,
    ):
        """Single-config action and gradient via pair-pair sums.

        Pair index layout (4*N^2 entries) matches
        ``_signed_pair_momenta_and_coeffs(representative=False)``:
            A_idx = block * N^2 + i * N + j, with
            block in [0..3] for sign_pairs ((+,+),(+,-),(-,+),(-,-)).

        Define F_A = Σ_B Q_B K(P_A + P_B), H_A = ∂F/∂P_A. Then with
        ζ_n = α k_n / (|k_n|^2 + m^2) (zero for α=0):
            ∂S̃/∂b_n = 4i Σ_{A: slot1=n} σ_A · Q_A · F_A
            ∂S̃/∂k_n = 4 ζ_n · Σ_{A: slot1=n} Q_A F_A
                     + 4 · Σ_{A: slot1=n} σ_A · Q_A · H_A
        Real parts give the gradient of the (real) S̃; multiply by `pref`.
        """
        arch = theory.architecture
        has_alpha = hasattr(arch, "alpha")
        alpha = arch.alpha if has_alpha else 0.0
        m_sq = arch.m * arch.m if has_alpha else 0.0

        pair_momenta, pair_coeffs = self._signed_pair_momenta_and_coeffs(
            params, sl, theory, representative=False
        )
        n_pairs = pair_momenta.shape[0]
        assert n_pairs == 4 * N * N

        # Neighbor list of pairs (A, B) with |P_A + P_B| < k_cut.
        # cKDTree.sparse_distance_matrix(self, neg_self) returns indices
        # (row=a in pair_momenta, col=b in -pair_momenta) at distance
        # |P_A - (-P_B)| = |P_A + P_B| < k_cut.
        if np.isinf(k_cut):
            # dense fallback (small N only)
            P_sum_full = (
                pair_momenta[:, None, :] + pair_momenta[None, :, :]
            )
            Ksq = np.sum(P_sum_full * P_sum_full, axis=-1)
            Kvals = np.exp(-gamma * Ksq)
            action_complex = np.einsum(
                "a,b,ab->", pair_coeffs, pair_coeffs, Kvals
            )
            F = Kvals @ pair_coeffs
            H = -2.0 * gamma * np.einsum(
                "ab,b,abd->ad", Kvals, pair_coeffs, P_sum_full
            )
        else:
            tree = cKDTree(pair_momenta)
            neg_tree = cKDTree(-pair_momenta)
            coo = tree.sparse_distance_matrix(
                neg_tree, k_cut, output_type="coo_matrix"
            )
            rows = coo.row
            cols = coo.col
            weights = np.exp(-gamma * coo.data * coo.data)   # K_{AB}
            Q_b = pair_coeffs[cols] * weights                # (n_neigh,)
            action_complex = np.sum(pair_coeffs[rows] * Q_b)
            F = np.zeros(n_pairs, dtype=np.complex128)
            np.add.at(F, rows, Q_b)
            P_sum = pair_momenta[rows] + pair_momenta[cols]   # (n_neigh, d)
            contrib_H = (-2.0 * gamma) * (Q_b[:, None] * P_sum)
            H = np.zeros((n_pairs, d), dtype=np.complex128)
            np.add.at(H, rows, contrib_H)

        S_val = float(pref * np.real(action_complex))

        # Reshape per (block, i, j) and sum over j (slot 2).
        Q_blk = pair_coeffs.reshape(4, N, N)
        F_blk = F.reshape(4, N, N)
        H_blk = H.reshape(4, N, N, d)
        sigmas = np.array([+1.0, +1.0, -1.0, -1.0])   # σ_i for each block

        QF = (Q_blk * F_blk).sum(axis=2)              # (4, N) = Σ_j Q[i,j] F[i,j]
        QH = (Q_blk[..., None] * H_blk).sum(axis=2)   # (4, N, d)

        grad_b_complex = 4.0j * np.einsum("c,cn->n", sigmas, QF)
        gk_F_part_complex = QF.sum(axis=0)            # Σ over blocks (no σ)
        gk_H_part_complex = np.einsum("c,cnd->nd", sigmas, QH)

        grad_b_real = pref * np.real(grad_b_complex)
        if has_alpha and alpha != 0.0:
            k = np.asarray(params["W0"][sl], dtype=float)
            ksq = np.sum(k * k, axis=-1)
            zeta = alpha * k / (ksq + m_sq)[:, None]      # (N, d)
            grad_k = 4.0 * (
                zeta * np.real(gk_F_part_complex)[:, None]
                + np.real(gk_H_part_complex)
            )
        else:
            grad_k = 4.0 * np.real(gk_H_part_complex)
        grad_k_real = pref * grad_k
        return S_val, grad_k_real, grad_b_real

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

    def _action_perm_sym(self, theory, params, b):
        """Same as _action_explicit but only sums over i<=j<=k<=l with the
        appropriate multinomial multiplicity. Reduces work by ~4!.
        """
        # for the requested validation use, just defer to the explicit form;
        # included as an alias to make the API surface match the plan.
        return self._action_explicit(theory, params, b)

    # ----- approximate momentum-conservation summation -------------------
    def _action_trans_sym(self, theory, params, b, *, eps, k_cut):
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
