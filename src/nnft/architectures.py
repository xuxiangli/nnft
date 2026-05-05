"""Single-neuron architectures and parameter distributions."""

from abc import ABC, abstractmethod

import numpy as np

from .analytics import f_Lambda, omega_alpha


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

    def log_pdf(self, x):
        """Log density at x. Override in subclasses; the adapter form falls back
        to the wrapped scipy frozen distribution's logpdf when available."""
        if hasattr(self._rv, "logpdf"):
            return np.asarray(self._rv.logpdf(np.asarray(x)))
        raise NotImplementedError(
            f"{type(self).__name__} has no log_pdf; override or wrap a scipy "
            f"frozen distribution"
        )


class Constant(Distribution):
    """Degenerate distribution returning a fixed scalar value."""

    def __init__(self, value):
        self.value = float(value)

    def sample(self, size, rng):
        return np.full(size, self.value)

    def log_pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x == self.value, 0.0, -np.inf)


class Uniform(Distribution):
    def __init__(self, low=0.0, high=1.0):
        """
        Args:
            low:  lower bound of the uniform interval (inclusive).
            high: upper bound of the uniform interval (exclusive).
        """
        self.low = float(low)
        self.high = float(high)
        self.length = self.high - self.low

    def sample(self, size, rng):
        return rng.uniform(self.low, self.high, size=size)

    def log_pdf(self, x):
        x = np.asarray(x, dtype=float)
        in_range = (x >= self.low) & (x <= self.high)
        return np.where(in_range, -np.log(self.length), -np.inf)


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

    def log_pdf(self, x):
        x = np.asarray(x, dtype=float)
        return -0.5 * ((x - self.mu) / self.sigma) ** 2 - 0.5 * np.log(
            2.0 * np.pi
        ) - np.log(self.sigma)


class UniBall(Distribution):
    def __init__(self, d=1, Lambda=1.0):
        """
        Args:
            d: dimension of the ball in which to sample points uniformly.
            Lambda: radius of the ball.
        """
        self.d = int(d)
        self.Lambda = float(Lambda)

    def sample(self, size, rng):
        n = size[0]
        x = rng.normal(size=(n, self.d + 2))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return self.Lambda * x[:, :self.d]


class UniSphere(Distribution):
    """Uniform distribution on the unit sphere S^{d-1} in R^d.

    Uses the isotropy of an i.i.d. standard Gaussian: drawing v ~ N(0, I_d) and
    returning v / ||v|| gives a unit vector whose direction is uniform over
    S^{d-1}.
    """

    def __init__(self, d):
        self.d = int(d)

    def sample(self, size, rng):
        n = size[0]
        v = rng.normal(size=(n, self.d))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v


class RegulatedMomentum(Distribution):
    """Sample k in R^d with p(k) propto f_Lambda(k^2) / (k^2 + m^2)^(alpha + 1).

    The distribution is radially symmetric: directions are drawn from a
    UniSphere, radii by inverse-CDF on a precomputed grid of the radial pdf
        p_r(r) propto r^(d-1) f_Lambda(r^2) / (r^2 + m^2)^(alpha + 1).
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
            r_max = self.Lambda * np.sqrt(2.0 * np.log(1.0 / self._GAUSS_TAIL_EPS))
        elif regulator == "none":
            r_max = np.inf
        else:
            raise ValueError(f"unknown regulator {regulator!r}")

        # Two-piece radial grid: first half densely covers [0, r_low] where
        # the pdf has most of its mass; second half spans [r_low, r_max].
        n_total = int(n_grid)
        n_low = n_total // 2
        n_high = n_total - n_low + 1   # +1 for the shared boundary point
        r_low = min(10.0 * self.m, r_max / 10.0)
        rs_low = np.linspace(0.0, r_low, n_low, endpoint=False)
        rs_high = np.linspace(r_low, r_max, n_high)
        rs = np.concatenate([rs_low, rs_high])
        radial_pdf = (
            rs ** (self.d - 1)
            * f_Lambda(rs * rs, self.Lambda, regulator)
            / (rs * rs + self.m * self.m) ** (self.alpha + 1.0)
        )
        cdf = np.concatenate(([0.0], np.cumsum(0.5 * (radial_pdf[1:] + radial_pdf[:-1]) * np.diff(rs))))
        cdf /= cdf[-1]
        self._cdf_grid = cdf
        self._r_grid = rs

    def _sample_radii(self, n, rng):
        u = rng.uniform(0.0, 1.0, size=n)
        return np.interp(u, self._cdf_grid, self._r_grid)

    def sample(self, size, rng):
        n = size[0]
        r = self._sample_radii(n, rng)
        directions = self._direction.sample((n,), rng)
        return r[:, None] * directions

    def log_pdf(self, x):
        """Unnormalized log-density of k in R^d up to a constant.

        log p(k) = log f_Lambda(k^2) - (alpha+1) log(k^2 + m^2) - log Z,
        Z = (2 pi)^d Omega_alpha (so that the marginalized k^2 sampler matches
        the closed-form Omega_alpha used by CosNetFT).
        """
        k = np.asarray(x, dtype=float)
        ksq = np.sum(k * k, axis=-1)
        if self.regulator == "hard":
            inside = ksq <= self.Lambda * self.Lambda
            log_freg = np.where(inside, 0.0, -np.inf)
        elif self.regulator == "gaussian":
            log_freg = -ksq / (2.0 * self.Lambda * self.Lambda)
        else:  # "none"
            log_freg = np.zeros_like(ksq)
        # closed-form Omega_alpha matches CosNetFT's normalization
        Omega = omega_alpha(self.d, self.m, self.alpha, self.Lambda, self.regulator)
        log_Z = self.d * np.log(2.0 * np.pi) + np.log(Omega)
        return log_freg - (self.alpha + 1.0) * np.log(ksq + self.m * self.m) - log_Z


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
    """varphi_j(x) = W1_j * tanh(W0_j . x + b0_j)."""

    def __init__(self, d_in):
        """
        Args:
            d_in: input dimension d. Each neuron's weight W0_j is a vector
                  in R^{d_in}; the bias b0_j is a scalar.
        """
        self.d_in = int(d_in)

    @property
    def param_spec(self):
        return {"W0": (self.d_in,), "b0": (), "W1": ()}

    def evaluate(self, x, params):
        W0 = params["W0"]                      # (N, d_in)
        b0 = params["b0"]                      # (N,)
        W1 = params["W1"]                      # (N,)
        pre = W0 @ x.T + b0[:, None]              # (N, n_pts)
        return W1[:, None] * np.tanh(pre)


class CosNet(Architecture):
    """varphi_j(x) = W1_j * cos(W0_j . x + b0_j)."""

    def __init__(self, d_in):
        """
        Args:
            d_in: input dimension d. Each neuron's weight W0_j is a vector
                  in R^{d_in}; the bias b0_j is a scalar phase.
        """
        self.d_in = int(d_in)

    @property
    def param_spec(self):
        return {"W0": (self.d_in,), "b0": (), "W1": ()}

    def evaluate(self, x, params):
        W0 = params["W0"]
        b0 = params["b0"]
        W1 = params["W1"]
        pre = W0 @ x.T + b0[:, None]
        return W1[:, None] * np.cos(pre)
    

class CosNetFT(Architecture):
    """Field-theoretic CosNet: varphi_j(x) = W1_j * cos(W0_j . x + b0_j) with

    W1_j *= (|W0_j|^2 + m^2)^(alpha / 2),

    and W0 ~ p(k) propto f_Lambda(k^2)/(k^2+m^2)^(alpha+1) (set by the
    matching default_dists()). Reproduces a free scalar 2-point function

        G^(2)(x1, x2) = int d^d k/(2 pi)^d  f_Lambda(k^2) / (k^2 + m^2) e^{i k . (x1 - x2)}.
    """

    def __init__(self, d_in, m, alpha, Lambda, regulator="hard"):
        """
        Args:
            d_in:      input dimension.
            m:         mass parameter.
            alpha:     power-law exponent in the W1 scaling and W0 PDF.
            Lambda:    UV cutoff scale.
            regulator: "hard" (Theta(Lambda^2 - k^2)) or "gaussian"
                       (exp(-k^2 / (2 Lambda^2))).
        """
        self.d_in = int(d_in)
        self.m = float(m)
        self.alpha = float(alpha)
        self.Lambda = float(Lambda)
        self.regulator = regulator
        self.Omega = omega_alpha(self.d_in, self.m, self.alpha, self.Lambda, regulator)

    @property
    def param_spec(self):
        return {"W0": (self.d_in,), "b0": (), "W1": ()}

    def evaluate(self, x, params):
        """Evaluate per-neuron outputs.
        
        Args:
        x: array (n_pts, d_in)
            params["W0"]: array(N_samples, d_in)
            params["b0"]: array(N_samples, )
            params["W1"]: array(N_samples, )

        Returns:
            array(N_samples, n_pts)
        """
        W0 = params["W0"]
        b0 = params["b0"]
        W1 = params["W1"] * (np.sum(W0**2, axis=-1) + self.m**2) ** (self.alpha / 2.0)
        pre = W0 @ x.T + b0[:, None]
        return W1[:, None] * np.cos(pre)

    def default_dists(self):
        """Canonical FT parameter distributions matching this architecture."""
        return {
            "W0": RegulatedMomentum(
                self.d_in, self.m, self.alpha, self.Lambda, self.regulator
            ),
            "b0": Uniform(-np.pi, np.pi),
            "W1": Constant(np.sqrt(2.0 * self.Omega)),
        }
