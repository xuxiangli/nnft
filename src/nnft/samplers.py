"""Parameter-space sampling strategies."""

from abc import ABC, abstractmethod

import numpy as np

from .architectures import Constant, Uniform


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

    Three proposal modes are provided:
        proposal_mode="all":    one MH step proposes a fresh full configuration
                                of N neurons; cost per step O(action eval).
        proposal_mode="single": one neuron is updated per inner step; one outer
                                step is N inner steps (a sweep). When the
                                interaction is a LambdaPhi4 with a real-space
                                quadrature method, a per-quadrature-point cache
                                makes each inner step O(M_x) instead of O(M_x N).
        proposal_mode="single_redraw": one neuron per inner step is redrawn
                                wholesale from the prior (independence proposal).
                                The prior cancels in the MH ratio, leaving
                                acceptance min(1, exp(-Delta S_int)). This is the
                                recommended mode for discrete-momentum priors
                                (finite box / LatticeMomentum), where local
                                proposals mix poorly; `proposals` is ignored.

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
        if proposal_mode not in ("all", "single", "single_redraw"):
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
            # Skip Constant (W1): its value is pinned, so log_pdf is 0 at the
            # state and -inf elsewhere; including it would spuriously zero a
            # meaningful -inf from other priors. Other priors' -inf (e.g. a
            # LatticeMomentum hop outside the UV cutoff, or a hard-cutoff
            # RegulatedMomentum) must propagate so _step_all rejects the move.
            if isinstance(dist, Constant):
                continue
            if hasattr(dist, "log_pdf"):
                total += float(np.sum(dist.log_pdf(state[name])))
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
        elif self.proposal_mode == "single_redraw":
            self._step_single_redraw_sweep(rng)
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
        if kind == "lattice":
            # Discrete lattice hop (finite box). Delegates to the parameter's
            # distribution, which knows the lattice spacing 2 pi / L. Symmetric.
            dist = self._theory.param_dists[name]
            return dist.propose(current, rng, scale), True
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

    def _step_single_redraw_sweep(self, rng):
        """One sweep of N single-neuron independence (prior-redraw) updates.

        For each neuron, draw (k, b) afresh from the prior. For a full prior
        redraw the prior densities cancel against the proposal densities in the
        MH ratio, so the acceptance is simply min(1, exp(-Delta S_int)). This
        mixes globally in the discrete momentum, unlike local lattice hops.
        Constant (W1) parameters are pinned and left unchanged.
        """
        N = self._theory.N
        order = rng.permutation(N)
        for i in order:
            self._step_single_redraw(int(i), rng)

    def _step_single_redraw(self, i, rng):
        spec = self._theory.architecture.param_spec
        new_neuron = {}
        for name, dist in self._theory.param_dists.items():
            if isinstance(dist, Constant):
                new_neuron[name] = self._state[name][i]
                continue
            new_neuron[name] = dist.sample((1,) + spec[name], rng)[0]

        trial = {k: v.copy() for k, v in self._state.items()}
        for name, val in new_neuron.items():
            trial[name][i] = val
        new_S = float(
            self.interaction.action(
                self._theory, trial, b=1,
                method=self.action_method,
                **self.action_kwargs,
            )[0]
        )
        # prior cancels for a full prior-redraw independence proposal
        log_alpha = -(new_S - self._S_int)
        self._n_propose += 1
        if np.log(rng.uniform()) < log_alpha:
            for name, val in new_neuron.items():
                self._state[name][i] = val
            self._S_int = new_S
            self._n_accept += 1

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
            # Skip Constant (W1): pinned value, log_pdf == 0 at the state. Other
            # priors' -inf must propagate so the move is rejected below.
            if isinstance(dist, Constant):
                continue
            if hasattr(dist, "log_pdf"):
                lp_old = dist.log_pdf(self._state[name][i])
                lp_new = dist.log_pdf(new_neuron[name])
                lp_old = np.sum(lp_old) if np.ndim(lp_old) > 0 else float(lp_old)
                lp_new = np.sum(lp_new) if np.ndim(lp_new) > 0 else float(lp_new)
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


class _GradientChainBase(Sampler):
    """Shared chain plumbing for gradient-based samplers (HMC, MALA).

    Maintains state (theta, S_int, log_prior, S_grad, lp_grad). For
    `action_method="real_space_mc"` a fixed quadrature `x_quad` is cached.
    For `action_method="trans_sym"` the action+grad are recomputed from
    scratch each call (no cached pair tree — pair momenta change with
    every k_i). Subclasses implement `_step(rng)`.

    log_prior / lp_grad EXCLUDE Uniform contributions: the Uniform on b0
    represents a periodic angle (cos/sin are 2pi-periodic) and the dynamics
    live on the universal cover. The Uniform support check is therefore
    skipped — leaving b0 unwrapped is consistent with the target density and
    avoids a discontinuity. Hard-cutoff Uniforms on k would be wrong here;
    use the Gaussian UV regulator.
    """

    _SUPPORTED_METHODS = ("real_space_mc", "trans_sym")

    def __init__(
        self,
        interaction,
        *,
        burn_in,
        thin,
        init_sampler,
        action_method,
        action_kwargs,
        resample_x_per_step,
        update_W1,
    ):
        if action_method not in self._SUPPORTED_METHODS:
            raise NotImplementedError(
                f"gradient-based samplers support {self._SUPPORTED_METHODS}, "
                f"got {action_method!r}"
            )
        self.interaction = interaction
        self.burn_in = int(burn_in)
        self.thin = int(thin)
        self.init_sampler = init_sampler if init_sampler is not None else IIDSampler()
        self.action_method = action_method
        self.action_kwargs = dict(action_kwargs) if action_kwargs else {}
        if action_method == "real_space_mc" and "M_x" not in self.action_kwargs:
            raise ValueError("action_kwargs must include M_x for real_space_mc")
        self.resample_x_per_step = bool(resample_x_per_step)
        self.update_W1 = bool(update_W1)

        self._theory = None
        self._state = None
        self._S_int = None
        self._log_prior = None
        self._x_quad = None
        self._S_grad = None
        self._lp_grad = None
        self._n_accept = 0
        self._n_propose = 0

    @property
    def acceptance_rate(self):
        if self._n_propose == 0:
            return float("nan")
        return self._n_accept / self._n_propose

    def _updated_names(self):
        out = []
        for name, dist in self._theory.param_dists.items():
            if isinstance(dist, Constant) and not self.update_W1:
                continue
            out.append(name)
        return out

    def _action_and_grad(self, state):
        flat = {k: v for k, v in state.items()}
        kw = {}
        if self.action_method == "real_space_mc":
            kw["x_quad"] = self._x_quad
        else:  # trans_sym
            for key in ("eps", "k_cut"):
                if key in self.action_kwargs:
                    kw[key] = self.action_kwargs[key]
        S, grads = self.interaction.action_and_grad(
            self._theory, flat, b=1, method=self.action_method, **kw,
        )
        return float(S[0]), grads

    def _log_prior_and_grad(self, state):
        """Sum log_pdf and gradient over distributions, skipping Uniform.

        Returns (-inf, ...) if any non-Uniform distribution puts the state
        outside its support (e.g. a hard-cutoff RegulatedMomentum).
        """
        total = 0.0
        grads = {}
        finite = True
        for name, dist in self._theory.param_dists.items():
            x = state[name]
            grads[name] = np.zeros_like(x, dtype=float)
            if isinstance(dist, Uniform):
                continue
            if isinstance(dist, Constant):
                continue
            lp = dist.log_pdf(x)
            if np.any(np.isneginf(lp)):
                finite = False
            else:
                total += float(np.sum(lp))
            if hasattr(dist, "grad_log_pdf"):
                g = dist.grad_log_pdf(x)
                grads[name] = np.asarray(g, dtype=float)
        if not finite:
            return -np.inf, grads
        return total, grads

    def _maybe_resample_xq(self, rng):
        if not self.resample_x_per_step:
            return
        if self.action_method != "real_space_mc":
            return
        M_x = self.action_kwargs["M_x"]
        self._x_quad = self.interaction.draw_mc_points(
            self._theory.architecture.d_in, int(M_x), rng
        )
        self._S_int, self._S_grad = self._action_and_grad(self._state)

    def _init_chain(self, theory, rng):
        N = theory.N
        spec = theory.architecture.param_spec
        params = self.init_sampler.sample(theory, 1, rng)
        self._state = {name: params[name].reshape((N,) + shape)
                       for name, shape in spec.items()}
        self._theory = theory
        if self.action_method == "real_space_mc":
            M_x = self.action_kwargs["M_x"]
            self._x_quad = self.interaction.draw_mc_points(
                theory.architecture.d_in, int(M_x), rng
            )
        self._S_int, self._S_grad = self._action_and_grad(self._state)
        self._log_prior, self._lp_grad = self._log_prior_and_grad(self._state)
        for _ in range(self.burn_in):
            self._step(rng)

    @abstractmethod
    def _step(self, rng):
        ...

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
                for (name, shape), arr in zip(spec.items(),
                                              [out[k] for k in spec])}


class HMCSampler(_GradientChainBase):
    """Hamiltonian Monte Carlo over (W0, b0).

    H(theta, p) = U(theta) + 0.5 * sum_n p_n^2 / mass_n,
    U(theta) = -log P_G(theta) + S_int(theta).

    `step_size` and `mass` are dicts keyed by parameter name (W0, b0). One
    full step = `n_leapfrog` leapfrog updates followed by a Metropolis
    accept/reject on the change in H. W1 is held fixed (CosNetFT pins it to
    sqrt(2 Omega_alpha)); the gradient through w_i = W1 (|k|^2 + m^2)^{a/2}
    enters via ∂S_int/∂W0 only.
    """

    def __init__(
        self,
        interaction,
        *,
        step_size,
        n_leapfrog,
        mass=None,
        burn_in=200,
        thin=1,
        init_sampler=None,
        action_method="real_space_mc",
        action_kwargs=None,
        resample_x_per_step=False,
        update_W1=False,
    ):
        super().__init__(
            interaction,
            burn_in=burn_in, thin=thin, init_sampler=init_sampler,
            action_method=action_method, action_kwargs=action_kwargs,
            resample_x_per_step=resample_x_per_step, update_W1=update_W1,
        )
        self.step_size = dict(step_size)
        self.n_leapfrog = int(n_leapfrog)
        self.mass = dict(mass) if mass else {}

    def _grad_U(self, S_grad, lp_grad, names):
        # ∇U = ∇S_int - ∇log P_G
        return {n: S_grad[n] - lp_grad[n] for n in names}

    def _step(self, rng):
        self._maybe_resample_xq(rng)
        names = self._updated_names()
        steps = {n: float(self.step_size[n]) for n in names}
        masses = {n: float(self.mass.get(n, 1.0)) for n in names}

        # snapshot for restore on reject
        state0 = {n: self._state[n].copy() for n in self._state}
        S0 = self._S_int
        lp0 = self._log_prior
        S_grad0 = {k: v.copy() for k, v in self._S_grad.items()}
        lp_grad0 = {k: v.copy() for k, v in self._lp_grad.items()}

        # draw momentum p ~ N(0, mass)
        p = {n: rng.normal(scale=np.sqrt(masses[n]),
                           size=self._state[n].shape) for n in names}
        KE0 = 0.5 * sum(np.sum(p[n] ** 2) / masses[n] for n in names)
        H0 = (-lp0 + S0) + KE0

        gradU = self._grad_U(self._S_grad, self._lp_grad, names)
        # half kick
        for n in names:
            p[n] = p[n] - 0.5 * steps[n] * gradU[n]

        ok = True
        for step_i in range(self.n_leapfrog):
            for n in names:
                self._state[n] = self._state[n] + steps[n] * p[n] / masses[n]
            self._log_prior, self._lp_grad = self._log_prior_and_grad(self._state)
            if not np.isfinite(self._log_prior):
                ok = False
                break
            self._S_int, self._S_grad = self._action_and_grad(self._state)
            gradU = self._grad_U(self._S_grad, self._lp_grad, names)
            kick = 0.5 if step_i == self.n_leapfrog - 1 else 1.0
            for n in names:
                p[n] = p[n] - kick * steps[n] * gradU[n]

        self._n_propose += 1
        if not ok:
            self._state = state0
            self._S_int = S0
            self._log_prior = lp0
            self._S_grad = S_grad0
            self._lp_grad = lp_grad0
            return

        KE_new = 0.5 * sum(np.sum(p[n] ** 2) / masses[n] for n in names)
        H_new = (-self._log_prior + self._S_int) + KE_new
        log_alpha = H0 - H_new
        if np.log(rng.uniform()) < log_alpha:
            self._n_accept += 1
        else:
            self._state = state0
            self._S_int = S0
            self._log_prior = lp0
            self._S_grad = S_grad0
            self._lp_grad = lp_grad0


class MALASampler(_GradientChainBase):
    """Metropolis-adjusted Langevin algorithm.

    Proposal: theta' = theta + tau * grad(log P) + sqrt(2 tau) * xi, xi ~ N(0, I).
    Asymmetric MH correction with q(theta'|theta) ∝ exp(-|theta'-theta-tau g|^2/(4 tau)).

    `tau` is a dict keyed by parameter name (per-name preconditioning);
    typical scales are tau_b ~ 1 for b0, tau_k ~ 1/Lambda^2 for W0.
    """

    def __init__(
        self,
        interaction,
        *,
        tau,
        burn_in=500,
        thin=1,
        init_sampler=None,
        action_method="real_space_mc",
        action_kwargs=None,
        resample_x_per_step=False,
        update_W1=False,
    ):
        super().__init__(
            interaction,
            burn_in=burn_in, thin=thin, init_sampler=init_sampler,
            action_method=action_method, action_kwargs=action_kwargs,
            resample_x_per_step=resample_x_per_step, update_W1=update_W1,
        )
        self.tau = dict(tau)

    def _grad_log_P(self, S_grad, lp_grad, names):
        # ∇log P = ∇log P_G - ∇S_int
        return {n: lp_grad[n] - S_grad[n] for n in names}

    def _step(self, rng):
        self._maybe_resample_xq(rng)
        names = self._updated_names()
        taus = {n: float(self.tau[n]) for n in names}

        glog = self._grad_log_P(self._S_grad, self._lp_grad, names)
        proposed = {k: v.copy() for k, v in self._state.items()}
        for n in names:
            xi = rng.normal(size=self._state[n].shape)
            proposed[n] = (
                self._state[n] + taus[n] * glog[n]
                + np.sqrt(2.0 * taus[n]) * xi
            )

        new_lp, new_lp_grad = self._log_prior_and_grad(proposed)
        self._n_propose += 1
        if not np.isfinite(new_lp):
            return
        new_S, new_S_grad = self._action_and_grad(proposed)
        new_glog = self._grad_log_P(new_S_grad, new_lp_grad, names)

        log_q_fwd = 0.0
        log_q_bwd = 0.0
        for n in names:
            d_fwd = proposed[n] - self._state[n] - taus[n] * glog[n]
            d_bwd = self._state[n] - proposed[n] - taus[n] * new_glog[n]
            log_q_fwd += -float(np.sum(d_fwd * d_fwd)) / (4.0 * taus[n])
            log_q_bwd += -float(np.sum(d_bwd * d_bwd)) / (4.0 * taus[n])

        log_alpha = (
            (new_lp - new_S) - (self._log_prior - self._S_int)
            + (log_q_bwd - log_q_fwd)
        )
        if np.log(rng.uniform()) < log_alpha:
            self._state = proposed
            self._S_int = new_S
            self._S_grad = new_S_grad
            self._log_prior = new_lp
            self._lp_grad = new_lp_grad
            self._n_accept += 1
