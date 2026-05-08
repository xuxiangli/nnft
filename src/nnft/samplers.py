"""Parameter-space sampling strategies."""

from abc import ABC, abstractmethod

import numpy as np


class Sampler(ABC):
    """Draws parameter realizations for a Theory."""

    @abstractmethod
    def sample(self, theory, n_samples, rng) -> dict:
        """Return one params dict with leading axis n_samples * theory.N.

        `params[name]` has shape `(n_samples * N, *spec_shape)`. The batch
        and neuron axes are flattened so the linear part of
        `architecture.evaluate` is a single 2D matmul.
        """
        ...


class IIDSampler(Sampler):
    """Each parameter drawn i.i.d. across (batch x neurons) from its Distribution."""

    def sample(self, theory, n_samples, rng):
        N = theory.N
        spec = theory.architecture.param_spec
        dists = theory.param_dists
        return {
            name: np.asarray(dists[name].sample((n_samples * N,) + shape, rng))
            for name, shape in spec.items()
        }


class MetropolisHastingsSampler(Sampler):
    """Metropolis-Hastings sampler for the modified PDF
        P(theta) propto P_G(theta) exp(-S_int(theta))
    over the per-neuron parameters.

    Two proposal modes are provided:
        proposal_mode="all":    one MH step proposes a fresh full configuration
                                of N neurons; cost per step O(action eval).
        proposal_mode="single": one neuron is updated per inner step; one outer
                                step is N inner steps (a sweep). When the
                                interaction is a LambdaPhi4 with a real-space
                                quadrature method, a per-quadrature-point cache
                                makes each inner step O(M_x) instead of O(M_x N).

    The chain state is held internally; successive calls to `sample` continue
    from where the previous call ended.
    """

    def __init__(
        self,
        interaction,
        proposals,
        *,
        proposal_mode="single",
        burn_in=1000,
        thin=1,
        init_sampler=None,
        action_method="real_space_mc",
        action_kwargs=None,
        resample_x_per_sweep=False,
    ):
        if proposal_mode not in ("all", "single"):
            raise ValueError(f"unknown proposal_mode {proposal_mode!r}")
        self.interaction = interaction
        self.proposals = dict(proposals)
        self.proposal_mode = proposal_mode
        self.burn_in = int(burn_in)
        self.thin = int(thin)
        self.init_sampler = init_sampler if init_sampler is not None else IIDSampler()
        self.action_method = action_method
        self.action_kwargs = dict(action_kwargs) if action_kwargs else {}
        self.resample_x_per_sweep = bool(resample_x_per_sweep)

        # Lazy chain state, materialized on first `sample` call.
        self._theory = None
        self._state = None         # dict name -> (N, *spec)
        self._S_int = None         # current interaction action
        self._log_prior = None     # current sum_n log P_G(theta_n)
        self._x_quad = None        # cached quadrature points (real-space methods)
        self._quad_weights = None
        self._Phi = None           # cached phi(x_q) for current state
        self._n_accept = 0
        self._n_propose = 0

    # ----- public ----------------------------------------------------------
    @property
    def acceptance_rate(self):
        if self._n_propose == 0:
            return float("nan")
        return self._n_accept / self._n_propose

    def sample(self, theory, n_samples, rng):
        spec = theory.architecture.param_spec
        if self._theory is not theory:
            self._init_chain(theory, rng)

        out = {name: np.empty((n_samples,) + (theory.N,) + shape, dtype=float)
               for name, shape in spec.items()}
        for s in range(n_samples):
            for _ in range(self.thin):
                self._step(rng)
            for name in spec:
                out[name][s] = self._state[name]
        return {name: arr.reshape((n_samples * theory.N,) + shape)
                for (name, shape), arr in zip(spec.items(), [out[k] for k in spec])}

    # ----- chain plumbing --------------------------------------------------
    def _init_chain(self, theory, rng):
        N = theory.N
        spec = theory.architecture.param_spec
        params = self.init_sampler.sample(theory, 1, rng)
        # reshape (N, *spec) -- 1 sample
        self._state = {name: params[name].reshape((N,) + shape)
                       for name, shape in spec.items()}
        self._theory = theory

        # Set up quadrature cache for real-space methods.
        if self.action_method == "real_space_mc":
            M_x = self.action_kwargs.get("M_x")
            if M_x is None:
                raise ValueError("action_kwargs must include M_x for real_space_mc")
            self._x_quad = self.interaction.draw_mc_points(
                theory.architecture.d_in, int(M_x), rng
            )
            self._quad_weights = None
        elif self.action_method == "real_space_hermite":
            n_h = self.action_kwargs.get("n_hermite")
            if n_h is None:
                raise ValueError(
                    "action_kwargs must include n_hermite for real_space_hermite"
                )
            self._x_quad, self._quad_weights = self.interaction.hermite_points(
                theory.architecture.d_in, int(n_h)
            )
        else:
            self._x_quad = None
            self._quad_weights = None

        self._Phi = self._compute_Phi(self._state)
        self._S_int = self._S_int_from_Phi(self._Phi)
        self._log_prior = self._compute_log_prior(self._state)

        # burn in
        for _ in range(self.burn_in):
            self._step(rng)

    def _flat_params(self, state):
        """Convert (N, *) state dict to the (1*N, *) flat layout."""
        return {k: v for k, v in state.items()}

    def _compute_Phi(self, state):
        """phi(x_q) for the current single configuration. Shape (n_q,).

        Falls back to None when no quadrature cache is in use (explicit/etc).
        """
        if self._x_quad is None:
            return None
        flat = self._flat_params(state)
        phi = self._theory.evaluate(self._x_quad, flat, b=1)   # (1, n_q)
        return phi[0]

    def _S_int_from_Phi(self, Phi):
        if self._x_quad is None:
            # full evaluation via interaction.action
            flat = self._flat_params(self._state)
            return float(
                self.interaction.action(
                    self._theory, flat, b=1,
                    method=self.action_method,
                    **self.action_kwargs,
                )[0]
            )
        d = self._theory.architecture.d_in
        if self.action_method == "real_space_mc":
            M_x = Phi.shape[0]
            prefactor = (
                self.interaction.lambda_
                * (2.0 * np.pi) ** (d / 2.0)
                * self.interaction.L ** d
                / (24.0 * M_x)
            )
            return float(prefactor * np.sum(Phi ** 4))
        if self.action_method == "real_space_hermite":
            return float(
                (self.interaction.lambda_ / 24.0)
                * np.sum(self._quad_weights * Phi ** 4)
            )
        raise AssertionError("unreachable")

    def _compute_log_prior(self, state):
        total = 0.0
        for name, dist in self._theory.param_dists.items():
            if hasattr(dist, "log_pdf"):
                lp = dist.log_pdf(state[name])
                # Constant returns -inf at every point that disagrees; in our
                # use case Constant draws exactly one value so log_pdf=0 there.
                lp = np.where(np.isneginf(lp), 0.0, lp) if isinstance(
                    lp, np.ndarray
                ) else lp
                total += float(np.sum(lp))
        return total

    def _maybe_resample_x_quad(self, rng):
        """Redraw real-space MC quadrature points and refresh cached Phi/S_int."""
        if not self.resample_x_per_sweep:
            return
        if self.action_method != "real_space_mc":
            return
        M_x = self.action_kwargs.get("M_x")
        if M_x is None:
            return
        self._x_quad = self.interaction.draw_mc_points(
            self._theory.architecture.d_in, int(M_x), rng
        )
        self._Phi = self._compute_Phi(self._state)
        self._S_int = self._S_int_from_Phi(self._Phi)

    # ----- one MH step -----------------------------------------------------
    def _step(self, rng):
        self._maybe_resample_x_quad(rng)
        if self.proposal_mode == "all":
            self._step_all(rng)
        else:
            self._step_single_sweep(rng)

    def _propose(self, name, current, rng):
        kind, scale = self.proposals.get(name, ("none", 0.0))
        if kind == "none":
            return current.copy(), True   # symmetric, identity
        if kind == "normal":
            return current + rng.normal(scale=scale, size=current.shape), True
        if kind == "uniform_wrap":
            new = current + rng.uniform(-scale, scale, size=current.shape)
            # wrap to [-pi, pi]
            new = ((new + np.pi) % (2.0 * np.pi)) - np.pi
            return new, True
        raise ValueError(f"unknown proposal kind {kind!r}")

    def _step_all(self, rng):
        """Propose fresh values for every neuron's parameters at once."""
        proposed = {}
        for name in self._state:
            proposed[name], _ = self._propose(name, self._state[name], rng)
        new_log_prior = self._compute_log_prior(proposed)
        if not np.isfinite(new_log_prior):
            self._n_propose += 1
            return
        new_Phi = self._compute_Phi(proposed) if self._x_quad is not None else None
        # compute proposed action
        if self._x_quad is None:
            flat = {k: v for k, v in proposed.items()}
            new_S = float(
                self.interaction.action(
                    self._theory, flat, b=1,
                    method=self.action_method,
                    **self.action_kwargs,
                )[0]
            )
        else:
            new_S = self._S_int_from_Phi_arr(new_Phi)
        log_alpha = (new_log_prior - self._log_prior) - (new_S - self._S_int)
        self._n_propose += 1
        if np.log(rng.uniform()) < log_alpha:
            self._state = proposed
            self._S_int = new_S
            self._log_prior = new_log_prior
            self._Phi = new_Phi
            self._n_accept += 1

    def _S_int_from_Phi_arr(self, Phi):
        d = self._theory.architecture.d_in
        if self.action_method == "real_space_mc":
            M_x = Phi.shape[0]
            prefactor = (
                self.interaction.lambda_
                * (2.0 * np.pi) ** (d / 2.0)
                * self.interaction.L ** d
                / (24.0 * M_x)
            )
            return float(prefactor * np.sum(Phi ** 4))
        if self.action_method == "real_space_hermite":
            return float(
                (self.interaction.lambda_ / 24.0)
                * np.sum(self._quad_weights * Phi ** 4)
            )
        raise AssertionError("unreachable")

    def _step_single_sweep(self, rng):
        """One sweep = N single-neuron MH proposals (random order)."""
        N = self._theory.N
        order = rng.permutation(N)
        for i in order:
            self._step_single(int(i), rng)

    def _step_single(self, i, rng):
        """Propose new (W0, b0, W1) for neuron i; cached Phi update if possible."""
        # propose
        new_neuron = {}
        for name, dist in self._theory.param_dists.items():
            cur = self._state[name][i]
            new_neuron[name], _ = self._propose(name, cur, rng)

        # prior delta
        log_prior_old = 0.0
        log_prior_new = 0.0
        for name, dist in self._theory.param_dists.items():
            if hasattr(dist, "log_pdf"):
                lp_old = dist.log_pdf(self._state[name][i])
                lp_new = dist.log_pdf(new_neuron[name])
                lp_old = np.sum(lp_old) if np.ndim(lp_old) > 0 else float(lp_old)
                lp_new = np.sum(lp_new) if np.ndim(lp_new) > 0 else float(lp_new)
                if np.isneginf(lp_old):
                    lp_old = 0.0
                log_prior_old += float(lp_old)
                log_prior_new += float(lp_new)
        if not np.isfinite(log_prior_new):
            self._n_propose += 1
            return

        # action via Phi cache when available
        if self._x_quad is not None:
            new_Phi = self._update_Phi_single(i, new_neuron)
            new_S = self._S_int_from_Phi_arr(new_Phi)
        else:
            # fall back: rebuild full state copy and call action()
            trial = {k: v.copy() for k, v in self._state.items()}
            for name, val in new_neuron.items():
                trial[name][i] = val
            flat = {k: v for k, v in trial.items()}
            new_S = float(
                self.interaction.action(
                    self._theory, flat, b=1,
                    method=self.action_method,
                    **self.action_kwargs,
                )[0]
            )
            new_Phi = None

        log_alpha = (log_prior_new - log_prior_old) - (new_S - self._S_int)
        self._n_propose += 1
        if np.log(rng.uniform()) < log_alpha:
            for name, val in new_neuron.items():
                self._state[name][i] = val
            self._S_int = new_S
            self._log_prior = self._log_prior + (log_prior_new - log_prior_old)
            if new_Phi is not None:
                self._Phi = new_Phi
            self._n_accept += 1

    def _update_Phi_single(self, i, new_neuron):
        """Recompute Phi by removing neuron i's old contribution and adding new."""
        theory = self._theory
        N = theory.N
        c_N = theory._c_N
        # old contribution at the cached x_quad
        old_state = {k: self._state[k][i:i + 1] for k in self._state}
        new_state = {k: new_neuron[k][None, ...] if np.ndim(new_neuron[k]) > 0
                     else np.asarray([new_neuron[k]]) for k in self._state}
        old_per_neuron = theory.architecture.evaluate(self._x_quad, old_state)  # (1, n_q)
        new_per_neuron = theory.architecture.evaluate(self._x_quad, new_state)  # (1, n_q)
        return self._Phi + c_N * (new_per_neuron[0] - old_per_neuron[0])
