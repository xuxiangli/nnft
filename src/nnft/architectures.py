"""Single-neuron architectures and parameter distributions."""

from abc import ABC, abstractmethod

import numpy as np


class Distribution:
    """Adapter around any object exposing rvs(size, random_state).

    scipy.stats frozen distributions satisfy this directly.
    """

    def __init__(self, rv):
        """
        Args:
            rv: any object with `rvs(size, random_state)` method (e.g. a
                scipy.stats frozen distribution). Used to draw parameter samples.
        """
        self._rv = rv

    def sample(self, size, rng):
        return self._rv.rvs(size=size, random_state=rng)


class Normal(Distribution):
    def __init__(self, mu=0.0, sigma=1.0):
        """
        Args:
            mu:    mean of the Gaussian.
            sigma: standard deviation of the Gaussian (must be > 0).
        """
        self.mu = float(mu)
        self.sigma = float(sigma)

    def sample(self, size, rng):
        return rng.normal(self.mu, self.sigma, size=size)


class Uniform(Distribution):
    def __init__(self, low=0.0, high=1.0):
        """
        Args:
            low:  lower bound of the uniform interval (inclusive).
            high: upper bound of the uniform interval (exclusive).
        """
        self.low = float(low)
        self.high = float(high)

    def sample(self, size, rng):
        return rng.uniform(self.low, self.high, size=size)


class Architecture(ABC):
    """Single-neuron architecture: defines varphi(x; theta_j)."""

    @property
    @abstractmethod
    def param_spec(self) -> dict:
        """Dict mapping parameter name -> per-neuron shape (tuple)."""
        ...

    @abstractmethod
    def evaluate(self, x, params):
        """Evaluate per-neuron outputs.

        x: array (n_pts, d_in)
        params: dict name -> array of shape (N, *spec_shape)
        returns: array (N, n_pts)
        """


class DenseTanh(Architecture):
    """varphi_j(x) = tanh(W_j . x + b_j), W_j in R^{d_in}, b_j in R."""

    def __init__(self, d_in):
        """
        Args:
            d_in: input dimension d. Each neuron's weight W_j is a vector
                  in R^{d_in}; the bias b_j is a scalar.
        """
        self.d_in = int(d_in)

    @property
    def param_spec(self):
        return {"W": (self.d_in,), "b": ()}

    def evaluate(self, x, params):
        W = params["W"]                       # (N, d_in)
        b = params["b"]                       # (N,)
        pre = W @ x.T + b[:, None]            # (N, n_pts)
        return np.tanh(pre)


class CosNet(Architecture):
    """varphi_j(x) = cos(W_j . x + b_j)."""

    def __init__(self, d_in):
        """
        Args:
            d_in: input dimension d. Each neuron's weight W_j is a vector
                  in R^{d_in}; the bias b_j is a scalar phase.
        """
        self.d_in = int(d_in)

    @property
    def param_spec(self):
        return {"W": (self.d_in,), "b": ()}

    def evaluate(self, x, params):
        W = params["W"]
        b = params["b"]
        pre = W @ x.T + b[:, None]
        return np.cos(pre)
