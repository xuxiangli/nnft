"""Theory class (JAX): fixed architecture + parameter PDFs.

Field: phi(x) = c_N * sum_{j=1..N} varphi(x; theta_j),
       c_N set by `normalization`.
"""

from math import factorial

import jax
import jax.numpy as jnp

from .samplers import IIDSampler


def _set_partitions(n):
    """Yield all set partitions of {0, ..., n-1} as lists of frozensets."""
    if n == 0:
        yield []
        return
    if n == 1:
        yield [frozenset({0})]
        return
    for rest in _set_partitions(n - 1):
        shifted = [frozenset(x + 1 for x in b) for b in rest]
        yield [frozenset({0})] + shifted
        for i in range(len(shifted)):
            new = list(shifted)
            new[i] = new[i] | {0}
            yield new


class Theory:
    _NORMALIZATIONS = {
        "1/sqrt(N)": lambda N: 1.0 / jnp.sqrt(float(N)),
        "1/N": lambda N: 1.0 / float(N),
        "none": lambda N: 1.0,
    }

    def __init__(self, architecture, N, param_dists, normalization="1/sqrt(N)"):
        """
        Args:
            architecture: Architecture instance defining varphi(x; theta_j).
            N:            int, number of neurons in the last layer.
            param_dists:  dict mapping each parameter name (matching
                          `architecture.param_spec`) to a Distribution
                          giving its per-neuron PDF in the i.i.d. baseline.
            normalization: prefactor c_N in phi = c_N * sum_j varphi.
                          One of "1/sqrt(N)", "1/N", "none".
        """
        spec = architecture.param_spec
        missing = set(spec) - set(param_dists)
        extra = set(param_dists) - set(spec)
        if missing or extra:
            raise ValueError(
                f"param_dists keys must match architecture.param_spec; "
                f"missing={missing}, extra={extra}"
            )
        if normalization not in self._NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {list(self._NORMALIZATIONS)}"
            )
        self._architecture = architecture
        self._N = int(N)
        self._param_dists = dict(param_dists)
        self._normalization = normalization
        self._c_N = self._NORMALIZATIONS[normalization](self._N)

    @property
    def architecture(self):
        return self._architecture

    @property
    def N(self):
        return self._N

    @property
    def param_dists(self):
        return self._param_dists

    @property
    def normalization(self):
        return self._normalization

    # ---- core ----
    def evaluate(self, x_points, params):
        """phi at x_points (shape (n_pts, d_in)) for one params dict."""
        x = jnp.atleast_2d(jnp.asarray(x_points, dtype=jnp.float32))
        per_neuron = self.architecture.evaluate(x, params)   # (N, n_pts)
        return self._c_N * per_neuron.sum(axis=0)            # (n_pts,)

    def log_prior(self, params):
        """Joint log-density of `params` under the i.i.d. baseline.

        Sums independent log-probs across all entries of every parameter.
        Override (or pass interdependent terms via a subclass) when the
        prior factorizes non-trivially; MetropolisHastingsSampler will
        target whatever this returns.
        """
        total = 0.0
        for name, dist in self._param_dists.items():
            total = total + jnp.sum(dist.log_prob(params[name]))
        return total

    def sample_field(self, x_points, n_samples, key, sampler=None):
        """Field values at multiple points, shape (n_samples, n_pts).

        One vmap'd evaluation per parameter draw.
        """
        sampler = sampler if sampler is not None else IIDSampler()
        x = jnp.atleast_2d(jnp.asarray(x_points, dtype=jnp.float32))
        params = sampler.sample(self, n_samples, key)
        eval_one = lambda p: self.evaluate(x, p)
        return jax.vmap(eval_one)(params)                    # (n_samples, n_pts)

    # ---- correlators ----
    def correlator(
        self,
        x_points,
        n_samples,
        key,
        sampler=None,
        connected=False,
        bootstrap=200,
    ):
        """Monte-Carlo estimate of G^(n)(x_1,...,x_n), n = len(x_points).

        Returns (value, stderr) as Python floats.
        - connected=False: ordinary moment, stderr from sample std / sqrt(n).
        - connected=True : cumulant via moments-to-cumulants formula;
                           stderr from `bootstrap` resamples.
        """
        samples = self.sample_field(x_points, n_samples, key, sampler=sampler)
        if not connected:
            prod = jnp.prod(samples, axis=1)
            value = float(prod.mean())
            stderr = float(jnp.std(prod, ddof=1) / jnp.sqrt(n_samples))
            return value, stderr

        value = float(_cumulant_from_samples(samples))
        boot_key = jax.random.fold_in(key, 0xB007)
        n = samples.shape[0]
        boot_keys = jax.random.split(boot_key, bootstrap)
        boot_vals = jnp.array(
            [
                _cumulant_from_samples(
                    samples[jax.random.randint(bk, (n,), 0, n)]
                )
                for bk in boot_keys
            ]
        )
        return value, float(jnp.std(boot_vals, ddof=1))


def _cumulant_from_samples(samples):
    """Connected n-point function via set-partition moments-to-cumulants.

    kappa_n = sum_{pi} (-1)^{|pi|-1} (|pi|-1)! prod_{B in pi} E[prod_{i in B} phi_i]
    """
    n = samples.shape[1]
    moments = {}
    total = 0.0
    for partition in _set_partitions(n):
        k = len(partition)
        sign = (-1) ** (k - 1) * factorial(k - 1)
        term = 1.0
        for block in partition:
            if block not in moments:
                cols = jnp.array(sorted(block))
                moments[block] = jnp.mean(jnp.prod(samples[:, cols], axis=1))
            term = term * moments[block]
        total = total + sign * term
    return total
