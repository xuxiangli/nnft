"""Parameter-space sampling strategies (JAX).

A Sampler returns parameter draws as a dict whose entries have leading
shape (n_draws, N, *spec). The Theory then `vmap`s field evaluation
over the n_draws axis.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import jax.random as jr


class Sampler(ABC):
    """Draws n_draws parameter realizations from the theory's prior."""

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


class MetropolisHastingsSampler(Sampler):
    """Random-walk Metropolis-Hastings on the joint parameter vector.

    Targets `theory.log_prior(params)`. Proposal is an isotropic Gaussian
    perturbation with scale `step_size` applied independently to every
    parameter component (across all neurons).

    Useful when the prior is *not* i.i.d. across neurons -- override
    `Theory.log_prior` with an interdependent log-density and this
    sampler will track it.
    """

    def __init__(self, step_size=0.1, n_warmup=1000, thin=1, init_sampler=None):
        """
        Args:
            step_size:    Gaussian proposal std.
            n_warmup:     burn-in iterations discarded before collecting samples.
            thin:         keep every `thin`-th post-warmup state.
            init_sampler: Sampler used to draw the chain's initial state
                          (default: IIDSampler so the chain starts from the
                          factorized prior).
        """
        self.step_size = float(step_size)
        self.n_warmup = int(n_warmup)
        self.thin = int(thin)
        self.init_sampler = init_sampler if init_sampler is not None else IIDSampler()
        self.last_acceptance = None

    def sample(self, theory, n_draws, key):
        key_init, key_warm, key_chain = jr.split(key, 3)

        init_batch = self.init_sampler.sample(theory, 1, key_init)
        init = {name: arr[0] for name, arr in init_batch.items()}

        log_prior = theory.log_prior
        names = tuple(init)
        # Per-name fold offsets so each parameter's proposal noise is
        # independent within a step. Static at trace time.
        name_offsets = {name: _name_hash(name) for name in names}
        step_size = self.step_size

        def step(carry, key_step):
            params, lp = carry
            k_prop_root, k_acc = jr.split(key_step)
            prop = {
                name: params[name]
                + step_size
                * jr.normal(
                    jr.fold_in(k_prop_root, name_offsets[name]), params[name].shape
                )
                for name in names
            }
            lp_prop = log_prior(prop)
            log_u = jnp.log(jr.uniform(k_acc))
            accept = log_u < (lp_prop - lp)
            new_params = {
                name: jnp.where(accept, prop[name], params[name]) for name in names
            }
            new_lp = jnp.where(accept, lp_prop, lp)
            return (new_params, new_lp), accept

        lp0 = log_prior(init)

        # Warmup pass: scan and discard.
        warm_keys = jr.split(key_warm, self.n_warmup)
        (state_warm, lp_warm), _ = jax.lax.scan(step, (init, lp0), warm_keys)

        # Collection pass: scan over n_draws*thin and keep every `thin`-th.
        n_chain = n_draws * self.thin
        chain_keys = jr.split(key_chain, n_chain)

        def collect_step(carry, key_step):
            new_carry, accepted = step(carry, key_step)
            return new_carry, (new_carry[0], accepted)

        _, (trajectory, accepts) = jax.lax.scan(
            collect_step, (state_warm, lp_warm), chain_keys
        )
        thinned = {
            name: arr[self.thin - 1 :: self.thin] for name, arr in trajectory.items()
        }
        self.last_acceptance = float(jnp.mean(accepts))
        return thinned


def _name_hash(name):
    """Stable small uint32 hash of a parameter name; static at trace time."""
    h = 0
    for c in name:
        h = (h * 131 + ord(c)) & 0xFFFFFFFF
    return h
