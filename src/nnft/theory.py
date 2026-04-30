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

    def __init__(self, architecture, N, param_dists, normalization="1/sqrt(N)"):
        """
        Args:
            architecture: Architecture instance defining the single-neuron
                          map varphi(x; theta_j). Determines which parameters
                          each neuron has and their per-neuron shapes.
            N:            int, number of neurons in the last (output) layer.
            param_dists:  dict mapping each parameter name (as declared in
                          `architecture.param_spec`) to a Distribution
                          giving its per-neuron PDF. Keys must match the
                          architecture's param_spec exactly.
            normalization: prefactor c_N in phi(x) = c_N * sum_j varphi(x;theta_j).
                          One of "1/sqrt(N)" (NNGP / standard NN init),
                          "1/N" (mean-field), or "none" (raw sum).
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
    def evaluate(self, x_points, params):
        """phi at x_points (shape (n_pts, d_in)) for given params dict."""
        x = np.atleast_2d(np.asarray(x_points, dtype=float))
        per_neuron = self.architecture.evaluate(x, params)   # (N, n_pts)
        return self._c_N * per_neuron.sum(axis=0)            # (n_pts,)

    def sample_field(self, x_points, n_samples, rng, sampler=None):
        """Field values at multiple points, shape (n_samples, n_pts).

        Each parameter draw produces values at all x_points simultaneously.
        """
        sampler = sampler if sampler is not None else IIDSampler()
        x = np.atleast_2d(np.asarray(x_points, dtype=float))
        out = np.empty((n_samples, x.shape[0]))
        for i, params in enumerate(sampler.sample(self, n_samples, rng)):
            out[i] = self.evaluate(x, params)
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
    ):
        """Monte-Carlo estimate of G^(n)(x_1,...,x_n), n = len(x_points).

        Returns (value, stderr).
        - connected=False: ordinary moment, stderr from sample std / sqrt(n).
        - connected=True : cumulant via moments-to-cumulants formula;
                           stderr from `bootstrap` resamples.
        """
        samples = self.sample_field(x_points, n_samples, rng, sampler=sampler)
        if not connected:
            prod = np.prod(samples, axis=1)
            return float(prod.mean()), float(prod.std(ddof=1) / np.sqrt(n_samples))
        value = _cumulant_from_samples(samples)
        # bootstrap stderr
        boot_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        vals = np.empty(bootstrap)
        for k in range(bootstrap):
            idx = boot_rng.integers(0, n_samples, size=n_samples)
            vals[k] = _cumulant_from_samples(samples[idx])
        return float(value), float(vals.std(ddof=1))


def _cumulant_from_samples(samples):
    """Connected n-point function from samples (n_samples, n) via set partitions.

    kappa_n = sum_{pi} (-1)^{|pi|-1} (|pi|-1)! prod_{B in pi} E[prod_{i in B} phi_i]
    """
    n = samples.shape[1]
    # cache moments E[prod_{i in B} phi_i] for each subset B (as frozenset).
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
