"""Theory class: fixed architecture + parameter PDFs; exposes correlator methods."""

from math import factorial

import numpy as np

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

        # element 0 in its own block
        yield [frozenset({0})] + shifted
        # element 0 joins an existing block
        for i in range(len(shifted)):
            new = list(shifted)
            new[i] = new[i] | {0}
            yield new


class Theory:
    """A fixed NN-FT theory: architecture + N + parameter distributions.

    Field: phi(x) = c_N * sum_{j=1..N} varphi(x; theta_j)
    where c_N is set by `normalization`.
    """

    _NORMALIZATIONS = {
        "1/sqrt(N)": lambda N: 1.0 / np.sqrt(N),
        "1/N": lambda N: 1.0 / N,
        "none": lambda N: 1.0,
    }

    def __init__(self, architecture, N, param_dists, normalization="1/sqrt(N)",
                 dtype=np.float64):
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
        self._dtype = np.dtype(dtype)

    # frozen attributes ---------------------------------------------------
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

    # core ----------------------------------------------------------------
    def evaluate(self, x_points, params, b=1):
        """Field values phi(x) at x_points for `b` parameter draws.

        Args:
            x_points: array (n_pts, d_in).
            params:   dict whose values have shape (b * N, *spec_shape).
            b:        number of parameter draws packed into `params`.

        Returns:
            array (b, n_pts).
        """
        dt = self._dtype
        x = np.atleast_2d(np.asarray(x_points, dtype=dt))
        params = {k: np.asarray(v, dtype=dt) for k, v in params.items()}
        per_neuron = self.architecture.evaluate(x, params)        # (b*N, n_pts)
        return self._c_N * per_neuron.reshape(b, self._N, -1).sum(axis=1)

    def sample_field(self, x_points, n_samples, rng, sampler=None, batch_size=None):
        """Field values at x_points, shape (n_samples, n_pts).

        `batch_size` controls how many field samples are drawn per call to
        `sampler.sample`; defaults to `n_samples` (a single call). Smaller
        batches trade speed for memory.
        """
        sampler = sampler if sampler is not None else IIDSampler()
        bs = int(batch_size) if batch_size is not None else int(n_samples)
        x = np.atleast_2d(np.asarray(x_points, dtype=self._dtype))
        out = np.empty((n_samples, x.shape[0]), dtype=self._dtype)
        done = 0
        while done < n_samples:
            b = min(bs, n_samples - done)
            params = sampler.sample(self, b, rng)
            out[done:done + b] = self.evaluate(x, params, b=b)
            done += b
        return out

    # correlators ---------------------------------------------------------
    def correlator(
        self,
        x_points,
        n_samples,
        rng,
        sampler=None,
        connected=False,
        bootstrap=200,
        n_configs=1,
        batch_size=None,
    ):
        """Monte-Carlo estimate of G^(n)(x_1,...,x_n).

        Disconnected mode (`connected=False`):
            x_points has shape (n_configs * n_corr, d_in). For each
            configuration ci in [0, n_configs), the n_corr query points
            x_points[ci*n_corr : (ci+1)*n_corr] define one G^(n) estimate.
            Returns (means, errors) shaped (n_configs,) — or scalars when
            n_configs == 1.
            Streams in chunks of `batch_size` field-samples to keep memory
            bounded; defaults to `n_samples`.

        Connected mode (`connected=True`): cumulant via moments-to-cumulants
            formula on `n_samples` materialised field samples; stderr from
            `bootstrap` resamples. Only `n_configs == 1` is supported here.
        """
        sampler = sampler if sampler is not None else IIDSampler()

        if connected:
            if n_configs != 1:
                raise ValueError("connected=True requires n_configs == 1")
            samples = self.sample_field(
                x_points, n_samples, rng, sampler=sampler, batch_size=batch_size
            )
            value = _cumulant_from_samples(samples)
            boot_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
            vals = np.empty(bootstrap)
            for k in range(bootstrap):
                idx = boot_rng.integers(0, n_samples, size=n_samples)
                vals[k] = _cumulant_from_samples(samples[idx])
            return float(value), float(vals.std(ddof=1))

        x = np.atleast_2d(np.asarray(x_points, dtype=float))
        n_pts = x.shape[0]
        if n_pts % n_configs != 0:
            raise ValueError(
                f"len(x_points)={n_pts} must be a multiple of n_configs={n_configs}"
            )
        n_corr = n_pts // n_configs
        bs = int(batch_size) if batch_size is not None else int(n_samples)

        sum_b = np.zeros(n_configs)
        sum_b_sq = np.zeros(n_configs)
        n_total = 0
        done = 0
        while done < n_samples:
            b = min(bs, n_samples - done)
            params = sampler.sample(self, b, rng)
            phi = self.evaluate(x, params, b=b)               # (b, n_configs * n_corr)
            prod = phi.reshape(b, n_configs, n_corr).prod(axis=2)  # (b, n_configs)
            # accumulate in float64 to keep precision over many batches
            sum_b += prod.sum(axis=0, dtype=np.float64)
            sum_b_sq += (prod * prod).sum(axis=0, dtype=np.float64)
            n_total += b
            done += b

        mean = sum_b / n_total
        # unbiased sample variance, then SE of the mean
        var = (sum_b_sq - n_total * mean * mean) / (n_total - 1)
        stderr = np.sqrt(np.maximum(var, 0.0) / n_total)

        if n_configs == 1:
            return float(mean[0]), float(stderr[0])
        return mean, stderr


def _cumulant_from_samples(samples):
    """Connected n-point function from samples (n_samples, n) via set partitions."""
    n = samples.shape[1]
    moments = {}
    total = 0.0

    for partition in _set_partitions(n):
        k = len(partition)
        sign = (-1) ** (k - 1) * factorial(k - 1)
        term = 1.0

        for block in partition:
            if block not in moments:
                cols = list(block)
                moments[block] = float(np.prod(samples[:, cols], axis=1).mean())
            term *= moments[block]

        total += sign * term
    return total
