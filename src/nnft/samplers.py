"""Parameter-space sampling strategies (JAX).

A Sampler returns parameter draws as a dict whose entries have leading
shape (n_draws, N, *spec). The Theory then `vmap`s field evaluation
over the n_draws axis.

MCMC samplers (Metropolis-Hastings, HMC, MALA) target `theory.log_density`
— which includes the interaction term when one is attached — and hold
atomic (Constant) parameters fixed at their initial value.
"""

from abc import ABC, abstractmethod

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr


class Sampler(ABC):
    """Draws n_draws parameter realizations."""

    @abstractmethod
    def sample(self, theory, n_draws, key) -> dict:
        """Returns dict[name -> array of shape (n_draws, N, *spec_shape)]."""
        ...


class IIDSampler(Sampler):
    """Each parameter drawn i.i.d. across neurons and across draws."""

    def sample(self, theory, n_draws, key):
        spec = theory.architecture.param_spec
        N = theory.N
        names = list(spec)
        keys = jr.split(key, len(names))
        out = {}
        for k, name in zip(keys, names):
            shape = (n_draws, N) + tuple(spec[name])
            out[name] = theory.param_dists[name].sample(k, shape)
        return out


# ---- helpers for MCMC --------------------------------------------------


def _split_atomic(theory, params):
    """Return (varying, fixed) dicts. `varying` holds non-atomic params
    (those subject to MCMC proposals); `fixed` holds atomic ones.
    """
    dists = theory.param_dists
    varying = {}
    fixed = {}
    for name, val in params.items():
        if getattr(dists[name], "is_atomic", False):
            fixed[name] = val
        else:
            varying[name] = val
    return varying, fixed


def _make_logdensity(theory, fixed):
    """Build a log-density fn over the varying-params dict, closing over
    the atomic params.
    """
    def logdensity_fn(varying):
        params = {**fixed, **varying}
        return theory.log_density(params)
    return logdensity_fn


# ---- Metropolis-Hastings ----------------------------------------------


class MetropolisHastingsSampler(Sampler):
    """Random-walk Metropolis-Hastings on the non-atomic parameters.

    Targets `theory.log_density`. Atomic parameters (e.g. CosNetFT's
    ``W1 ~ Constant``) are held at their initial value.
    """

    def __init__(self, step_size=0.1, n_warmup=1000, thin=1, init_sampler=None):
        self.step_size = float(step_size)
        self.n_warmup = int(n_warmup)
        self.thin = int(thin)
        self.init_sampler = (
            init_sampler if init_sampler is not None else IIDSampler()
        )
        self.last_acceptance = None

    def sample(self, theory, n_draws, key):
        key_init, key_warm, key_chain = jr.split(key, 3)

        init_batch = self.init_sampler.sample(theory, 1, key_init)
        init_full = {name: arr[0] for name, arr in init_batch.items()}
        init_varying, fixed = _split_atomic(theory, init_full)

        logdensity_fn = _make_logdensity(theory, fixed)
        names = tuple(init_varying)
        name_offsets = {name: _name_hash(name) for name in names}
        step_size = self.step_size

        def step(carry, key_step):
            params, lp = carry
            k_prop_root, k_acc = jr.split(key_step)
            prop = {
                name: params[name]
                + step_size
                * jr.normal(
                    jr.fold_in(k_prop_root, name_offsets[name]),
                    params[name].shape,
                )
                for name in names
            }
            lp_prop = logdensity_fn(prop)
            log_u = jnp.log(jr.uniform(k_acc))
            accept = log_u < (lp_prop - lp)
            new_params = {
                name: jnp.where(accept, prop[name], params[name])
                for name in names
            }
            new_lp = jnp.where(accept, lp_prop, lp)
            return (new_params, new_lp), accept

        lp0 = logdensity_fn(init_varying)

        warm_keys = jr.split(key_warm, self.n_warmup)
        (state_warm, lp_warm), _ = jax.lax.scan(
            step, (init_varying, lp0), warm_keys
        )

        n_chain = n_draws * self.thin
        chain_keys = jr.split(key_chain, n_chain)

        def collect_step(carry, key_step):
            new_carry, accepted = step(carry, key_step)
            return new_carry, (new_carry[0], accepted)

        _, (trajectory, accepts) = jax.lax.scan(
            collect_step, (state_warm, lp_warm), chain_keys
        )
        thinned = {
            name: arr[self.thin - 1 :: self.thin]
            for name, arr in trajectory.items()
        }
        # Re-attach fixed params (broadcast across draws).
        for name, val in fixed.items():
            thinned[name] = jnp.broadcast_to(
                val, (n_draws,) + val.shape
            )
        self.last_acceptance = float(jnp.mean(accepts))
        return thinned


# ---- blackjax wrappers -------------------------------------------------


def _run_blackjax_kernel(
    kernel_step, initial_state, key, n_warmup, n_collect, thin
):
    """Run a blackjax `step` kernel for `n_warmup + n_collect*thin` steps,
    returning the trajectory of *positions* (length n_collect) and the
    fraction of steps reported as accepted by the kernel info.

    `kernel_step(rng_key, state) -> (new_state, info)` follows the
    blackjax convention. `info.acceptance_rate` (HMC) or
    `info.acceptance_rate` (MALA) drives the diagnostic.
    """
    key_warm, key_chain = jr.split(key)

    def warmup_step(state, k):
        new_state, _ = kernel_step(k, state)
        return new_state, None

    warm_keys = jr.split(key_warm, n_warmup)
    state_after_warm, _ = jax.lax.scan(warmup_step, initial_state, warm_keys)

    def chain_step(state, k):
        new_state, info = kernel_step(k, state)
        return new_state, (new_state.position, info.acceptance_rate)

    chain_keys = jr.split(key_chain, n_collect * thin)
    _, (positions, accepts) = jax.lax.scan(
        chain_step, state_after_warm, chain_keys
    )
    # Thinning: take every `thin`-th from the collected positions
    positions = jax.tree_util.tree_map(
        lambda arr: arr[thin - 1 :: thin], positions
    )
    accepts_thin = accepts[thin - 1 :: thin]
    return positions, accepts_thin


class HMCSampler(Sampler):
    """Hamiltonian Monte Carlo via blackjax.

    Targets `theory.log_density`. Atomic params (Constant) are held
    fixed. The mass matrix is identity (no warm-up adaptation); tune
    `step_size` and `num_integration_steps` for the problem.
    """

    def __init__(
        self,
        step_size=0.01,
        num_integration_steps=20,
        n_warmup=500,
        thin=1,
        init_sampler=None,
    ):
        self.step_size = float(step_size)
        self.num_integration_steps = int(num_integration_steps)
        self.n_warmup = int(n_warmup)
        self.thin = int(thin)
        self.init_sampler = (
            init_sampler if init_sampler is not None else IIDSampler()
        )
        self.last_acceptance = None

    def sample(self, theory, n_draws, key):
        key_init, key_run = jr.split(key)
        init_batch = self.init_sampler.sample(theory, 1, key_init)
        init_full = {name: arr[0] for name, arr in init_batch.items()}
        init_varying, fixed = _split_atomic(theory, init_full)
        logdensity_fn = _make_logdensity(theory, fixed)

        inverse_mass_matrix = jax.tree_util.tree_map(
            lambda x: jnp.ones_like(x), init_varying
        )
        # blackjax HMC expects a flat IMM; build via jax.flatten_util.
        flat_imm, _ = jax.flatten_util.ravel_pytree(inverse_mass_matrix)

        hmc = blackjax.hmc(
            logdensity_fn,
            step_size=self.step_size,
            inverse_mass_matrix=flat_imm,
            num_integration_steps=self.num_integration_steps,
        )
        initial_state = hmc.init(init_varying)

        positions, accepts = _run_blackjax_kernel(
            hmc.step,
            initial_state,
            key_run,
            n_warmup=self.n_warmup,
            n_collect=n_draws,
            thin=self.thin,
        )
        for name, val in fixed.items():
            positions[name] = jnp.broadcast_to(val, (n_draws,) + val.shape)
        self.last_acceptance = float(jnp.mean(accepts))
        return positions


class MALASampler(Sampler):
    """Metropolis-Adjusted Langevin via blackjax.

    Targets `theory.log_density`. Atomic params (Constant) are held fixed.
    """

    def __init__(self, step_size=1e-3, n_warmup=1000, thin=1, init_sampler=None):
        self.step_size = float(step_size)
        self.n_warmup = int(n_warmup)
        self.thin = int(thin)
        self.init_sampler = (
            init_sampler if init_sampler is not None else IIDSampler()
        )
        self.last_acceptance = None

    def sample(self, theory, n_draws, key):
        key_init, key_run = jr.split(key)
        init_batch = self.init_sampler.sample(theory, 1, key_init)
        init_full = {name: arr[0] for name, arr in init_batch.items()}
        init_varying, fixed = _split_atomic(theory, init_full)
        logdensity_fn = _make_logdensity(theory, fixed)

        mala = blackjax.mala(logdensity_fn, step_size=self.step_size)
        initial_state = mala.init(init_varying)

        positions, accepts = _run_blackjax_kernel(
            mala.step,
            initial_state,
            key_run,
            n_warmup=self.n_warmup,
            n_collect=n_draws,
            thin=self.thin,
        )
        for name, val in fixed.items():
            positions[name] = jnp.broadcast_to(val, (n_draws,) + val.shape)
        self.last_acceptance = float(jnp.mean(accepts))
        return positions


def _name_hash(name):
    """Stable small uint32 hash; static at trace time."""
    h = 0
    for c in name:
        h = (h * 131 + ord(c)) & 0xFFFFFFFF
    return h
