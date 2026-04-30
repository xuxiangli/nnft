"""Parameter-space sampling strategies."""

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class Sampler(ABC):
    """Draws parameter realizations for a Theory."""

    @abstractmethod
    def sample(self, theory, n_draws, rng) -> Iterator[dict]:
        """Yield n_draws parameter dicts; each entry has shape (N, *spec_shape)."""
        ...


class IIDSampler(Sampler):
    """Each parameter drawn i.i.d. across neurons from its Distribution."""

    def sample(self, theory, n_draws, rng):
        N = theory.N
        spec = theory.architecture.param_spec
        dists = theory.param_dists
        for _ in range(n_draws):
            params = {}
            for name, shape in spec.items():
                params[name] = np.asarray(dists[name].sample((N,) + shape, rng))
            yield params
