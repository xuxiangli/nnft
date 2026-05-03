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
    """Each parameter drawn i.i.d. across (batch x neurons) from its Distribution.

    Stateless: a single `sample(theory, n_samples, rng)` call returns one
    params dict containing `n_samples * theory.N` independent draws per
    parameter. Iterating to a larger total is the caller's job (e.g.
    `Theory.correlator` chunks `M` into pieces of size `batch_size` and
    calls this once per chunk).
    """

    def sample(self, theory, n_samples, rng):
        N = theory.N
        spec = theory.architecture.param_spec
        dists = theory.param_dists
        return {
            name: np.asarray(dists[name].sample((n_samples * N,) + shape, rng))
            for name, shape in spec.items()
        }
