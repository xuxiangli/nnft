"""Single-neuron architectures and parameter distributions (JAX)."""

from abc import ABC, abstractmethod

import numpy as np
import jax.numpy as jnp
import jax.random as jr

from .analytics import f_Lambda, omega_alpha


class Distribution(ABC):
    """Per-parameter PDF for the i.i.d. baseline.

    All distributions expose:
        sample(key, shape) -> jnp.ndarray of `shape`
        log_prob(x)        -> jnp.ndarray (element-wise log-density)

    `is_atomic = True` marks a degenerate (point-mass) distribution; MCMC
    samplers use this to skip proposals on parameters that are not
    actually random.
    """

    is_atomic = False

    @abstractmethod
    def sample(self, key, shape) -> jnp.ndarray:
        ...

    @abstractmethod
    def log_prob(self, x) -> jnp.ndarray:
        ...


class Constant(Distribution):
    """Degenerate distribution returning a fixed scalar value."""

    is_atomic = True

    def __init__(self, value):
        self.value = float(value)

    def sample(self, key, shape):
        del key
        return jnp.full(shape, self.value)

    def log_prob(self, x):
        x = jnp.asarray(x)
        return jnp.where(x == self.value, 0.0, -jnp.inf)


class Uniform(Distribution):
    def __init__(self, low=0.0, high=1.0):
        self.low = float(low)
        self.high = float(high)
        self.length = self.high - self.low

    def sample(self, key, shape):
        return jr.uniform(key, shape, minval=self.low, maxval=self.high)

    def log_prob(self, x):
        x = jnp.asarray(x)
        in_range = (x >= self.low) & (x <= self.high)
        return jnp.where(in_range, -jnp.log(self.length), -jnp.inf)


class Normal(Distribution):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = float(mu)
        self.sigma = float(sigma)

    def sample(self, key, shape):
        return self.mu + self.sigma * jr.normal(key, shape)

    def log_prob(self, x):
        z = (jnp.asarray(x) - self.mu) / self.sigma
        return -0.5 * z * z - jnp.log(self.sigma) - 0.5 * jnp.log(2.0 * jnp.pi)


class UniBall(Distribution):
    """Uniform on the ball of radius Lambda in R^d."""

    def __init__(self, d=1, Lambda=1.0):
        self.d = int(d)
        self.Lambda = float(Lambda)

    def sample(self, key, shape):
        if len(shape) == 1:
            batch = (shape[0],)
        else:
            assert shape[-1] == self.d, (
                f"UniBall expected last dim {self.d}, got {shape}"
            )
            batch = shape[:-1]
        v = jr.normal(key, batch + (self.d + 2,))
        v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
        return self.Lambda * v[..., : self.d]

    def log_prob(self, x):
        from math import lgamma
        x = jnp.asarray(x)
        rsq = jnp.asarray(jnp.sum(x * x, axis=-1))
        inside = rsq <= self.Lambda * self.Lambda
        log_vol = (
            0.5 * self.d * np.log(np.pi)
            + self.d * np.log(self.Lambda)
            - lgamma(self.d / 2.0 + 1.0)
        )
        return jnp.where(inside, -log_vol, -jnp.inf)


class UniSphere(Distribution):
    """Uniform on the unit sphere S^{d-1}."""

    def __init__(self, d):
        self.d = int(d)

    def sample(self, key, shape):
        if len(shape) == 1:
            batch = (shape[0],)
        else:
            assert shape[-1] == self.d
            batch = shape[:-1]
        v = jr.normal(key, batch + (self.d,))
        return v / jnp.linalg.norm(v, axis=-1, keepdims=True)

    def log_prob(self, x):
        # Singular w.r.t. d-dim Lebesgue; not used by samplers.
        return jnp.zeros(jnp.asarray(x).shape[:-1])


class RegulatedMomentum(Distribution):
    """k in R^d with p(k) propto f_Lambda(k^2) / (k^2 + m^2)^(alpha+1).

    Radially symmetric: directions ~ UniSphere; radii by inverse-CDF on a
    precomputed grid. Grid is built with NumPy via analytics.f_Lambda /
    omega_alpha, then frozen as jnp constants.
    """

    _GAUSS_TAIL_EPS = 1e-14

    def __init__(self, d, m, alpha, Lambda, regulator="hard", n_grid=4096):
        self.d = int(d)
        self.m = float(m)
        self.alpha = float(alpha)
        self.Lambda = float(Lambda)
        self.regulator = regulator
        self._direction = UniSphere(self.d)

        if regulator == "hard":
            r_max = self.Lambda
        elif regulator == "gaussian":
            r_max = self.Lambda * np.sqrt(
                2.0 * np.log(1.0 / self._GAUSS_TAIL_EPS)
            )
        elif regulator == "none":
            r_max = np.inf
        else:
            raise ValueError(f"unknown regulator {regulator!r}")

        n_total = int(n_grid)
        n_low = n_total // 2
        n_high = n_total - n_low + 1
        r_low = min(10.0 * self.m, r_max / 10.0)
        rs_low = np.linspace(0.0, r_low, n_low, endpoint=False)
        rs_high = np.linspace(r_low, r_max, n_high)
        rs = np.concatenate([rs_low, rs_high])
        radial_pdf = (
            rs ** (self.d - 1)
            * f_Lambda(rs * rs, self.Lambda, regulator)
            / (rs * rs + self.m * self.m) ** (self.alpha + 1.0)
        )
        cdf = np.concatenate(
            (
                [0.0],
                np.cumsum(
                    0.5 * (radial_pdf[1:] + radial_pdf[:-1]) * np.diff(rs)
                ),
            )
        )
        cdf /= cdf[-1]
        self._cdf_grid = jnp.asarray(cdf)
        self._r_grid = jnp.asarray(rs)

        Omega = omega_alpha(
            self.d, self.m, self.alpha, self.Lambda, self.regulator
        )
        self._log_Z = float(self.d * np.log(2.0 * np.pi) + np.log(Omega))

    def sample(self, key, shape):
        if len(shape) == 1:
            batch = (shape[0],)
        else:
            assert shape[-1] == self.d
            batch = shape[:-1]
        n = int(np.prod(batch)) if batch else 1
        key_r, key_dir = jr.split(key)
        u = jr.uniform(key_r, (n,))
        r = jnp.interp(u, self._cdf_grid, self._r_grid)
        directions = self._direction.sample(key_dir, (n, self.d))
        out = r[:, None] * directions
        return out.reshape(batch + (self.d,))

    def log_prob(self, x):
        """Unnormalized log-density of k in R^d up to a constant.

        Returns shape = x.shape[:-1] (scalar per d-vector).
        """
        k = jnp.asarray(x)
        ksq = jnp.sum(k * k, axis=-1)
        if self.regulator == "hard":
            inside = ksq <= self.Lambda * self.Lambda
            log_freg = jnp.where(inside, 0.0, -jnp.inf)
        elif self.regulator == "gaussian":
            log_freg = -ksq / (2.0 * self.Lambda * self.Lambda)
        else:  # "none"
            log_freg = jnp.zeros_like(ksq)
        return (
            log_freg
            - (self.alpha + 1.0) * jnp.log(ksq + self.m * self.m)
            - self._log_Z
        )


# ---- Architectures --------------------------------------------------------


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
    """varphi_j(x) = W1_j * tanh(W0_j . x + b0_j)."""

    def __init__(self, d_in):
        self.d_in = int(d_in)

    @property
    def param_spec(self):
        return {"W0": (self.d_in,), "b0": (), "W1": ()}

    def evaluate(self, x, params):
        W0 = params["W0"]
        b0 = params["b0"]
        W1 = params["W1"]
        pre = W0 @ x.T + b0[:, None]
        return W1[:, None] * jnp.tanh(pre)


class CosNet(Architecture):
    """varphi_j(x) = W1_j * cos(W0_j . x + b0_j)."""

    def __init__(self, d_in):
        self.d_in = int(d_in)

    @property
    def param_spec(self):
        return {"W0": (self.d_in,), "b0": (), "W1": ()}

    def evaluate(self, x, params):
        W0 = params["W0"]
        b0 = params["b0"]
        W1 = params["W1"]
        pre = W0 @ x.T + b0[:, None]
        return W1[:, None] * jnp.cos(pre)


class CosNetFT(Architecture):
    """Field-theoretic CosNet.

    varphi_j(x) = W1_j * (|W0_j|^2 + m^2)^(alpha/2) * cos(W0_j . x + b0_j).

    The effective output weight is a deterministic function of W0
    (W1 ~ Constant in `default_dists`), so MCMC samplers update W0/b0
    and leave W1 fixed.
    """

    def __init__(self, d_in, m, alpha, Lambda, regulator="hard"):
        self.d_in = int(d_in)
        self.m = float(m)
        self.alpha = float(alpha)
        self.Lambda = float(Lambda)
        self.regulator = regulator
        self.Omega = omega_alpha(
            self.d_in, self.m, self.alpha, self.Lambda, regulator
        )

    @property
    def param_spec(self):
        return {"W0": (self.d_in,), "b0": (), "W1": ()}

    def evaluate(self, x, params):
        W0 = params["W0"]
        b0 = params["b0"]
        scale = (jnp.sum(W0 ** 2, axis=-1) + self.m ** 2) ** (self.alpha / 2.0)
        W1 = params["W1"] * scale
        pre = W0 @ x.T + b0[:, None]
        return W1[:, None] * jnp.cos(pre)

    def default_dists(self):
        return {
            "W0": RegulatedMomentum(
                self.d_in, self.m, self.alpha, self.Lambda, self.regulator
            ),
            "b0": Uniform(-np.pi, np.pi),
            "W1": Constant(np.sqrt(2.0 * self.Omega)),
        }
