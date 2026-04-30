"""Single-neuron architectures and parameter distributions (JAX)."""

from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax.random as jr


class Distribution(ABC):
    """Per-parameter PDF. Must support sampling and log-prob evaluation.

    log_prob is required so MCMC samplers can score proposals against
    an i.i.d. baseline prior.
    """

    @abstractmethod
    def sample(self, key, shape):
        """Draw values of given shape from this distribution.

        Args:
            key:   jax.random.PRNGKey.
            shape: tuple, output shape.
        """
        ...

    @abstractmethod
    def log_prob(self, x):
        """Element-wise log-density at x. Returns same shape as x."""
        ...


class Normal(Distribution):
    def __init__(self, mu=0.0, sigma=1.0):
        """
        Args:
            mu:    mean of the Gaussian.
            sigma: standard deviation (must be > 0).
        """
        self.mu = float(mu)
        self.sigma = float(sigma)

    def sample(self, key, shape):
        return self.mu + self.sigma * jr.normal(key, shape)

    def log_prob(self, x):
        z = (x - self.mu) / self.sigma
        return -0.5 * z * z - jnp.log(self.sigma) - 0.5 * jnp.log(2 * jnp.pi)


class Uniform(Distribution):
    def __init__(self, low=0.0, high=1.0):
        """
        Args:
            low:  lower bound (inclusive).
            high: upper bound (exclusive).
        """
        self.low = float(low)
        self.high = float(high)

    def sample(self, key, shape):
        return jr.uniform(key, shape, minval=self.low, maxval=self.high)

    def log_prob(self, x):
        in_support = (x >= self.low) & (x < self.high)
        width = self.high - self.low
        return jnp.where(in_support, -jnp.log(width), -jnp.inf)


class Architecture(ABC):
    """Single-neuron architecture: defines varphi(x; theta_j)."""

    @property
    @abstractmethod
    def param_spec(self) -> dict:
        """Dict mapping parameter name -> per-neuron shape (tuple)."""
        ...

    @abstractmethod
    def evaluate(self, x, params):
        """Per-neuron outputs.

        x:      array (n_pts, d_in)
        params: dict name -> array of shape (N, *spec_shape)
        returns: array (N, n_pts)
        """


class DenseTanh(Architecture):
    """varphi_j(x) = tanh(W_j . x + b_j)."""

    def __init__(self, d_in):
        """
        Args:
            d_in: input dimension. W_j has shape (d_in,); b_j is scalar.
        """
        self.d_in = int(d_in)

    @property
    def param_spec(self):
        return {"W": (self.d_in,), "b": ()}

    def evaluate(self, x, params):
        pre = params["W"] @ x.T + params["b"][:, None]
        return jnp.tanh(pre)


class CosNet(Architecture):
    """varphi_j(x) = cos(W_j . x + b_j)."""

    def __init__(self, d_in):
        """
        Args:
            d_in: input dimension. W_j has shape (d_in,); b_j is scalar phase.
        """
        self.d_in = int(d_in)

    @property
    def param_spec(self):
        return {"W": (self.d_in,), "b": ()}

    def evaluate(self, x, params):
        pre = params["W"] @ x.T + params["b"][:, None]
        return jnp.cos(pre)
