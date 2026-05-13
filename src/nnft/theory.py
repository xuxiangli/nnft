"""Theory class (JAX): fixed architecture + parameter PDFs.

Field: phi(x) = c_N * sum_{j=1..N} varphi(x; theta_j),
       c_N set by `normalization`.
"""

from math import factorial

import jax
import jax.numpy as jnp
import jax.random as jr
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

    def __init__(
        self,
        architecture,
        N,
        param_dists,
        normalization="1/sqrt(N)",
        interaction=None,
        action_method="trans_sym",
        action_kwargs=None,
    ):
        """
        Args:
            architecture: Architecture instance defining varphi(x; theta_j).
            N:            int, number of neurons in the last layer.
            param_dists:  dict matching architecture.param_spec, values are
                          Distribution instances giving the i.i.d. baseline.
            normalization: prefactor c_N in phi = c_N * sum_j varphi.
                          One of "1/sqrt(N)", "1/N", "none".
            interaction:  optional LambdaPhi4-like object. If provided,
                          `log_density(params) = log_prior - interaction.action(...)`.
            action_method: method= passed to interaction.action.
            action_kwargs: extra kwargs forwarded to interaction.action
                          (e.g. {"n_hermite": 6} or cached x_quad/quad_weights).
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
        self._interaction = interaction
        self._action_method = action_method
        self._action_kwargs = dict(action_kwargs) if action_kwargs else {}

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

    @property
    def interaction(self):
        return self._interaction

    # ---- core ----
    def evaluate(self, x_points, params):
        """phi at x_points (shape (n_pts, d_in)) for one params dict."""
        x = jnp.atleast_2d(jnp.asarray(x_points, dtype=jnp.float32))
        per_neuron = self.architecture.evaluate(x, params)   # (N, n_pts)
        return self._c_N * per_neuron.sum(axis=0)            # (n_pts,)

    def log_prior(self, params):
        """Joint log-density of `params` under the i.i.d. baseline."""
        total = 0.0
        for name, dist in self._param_dists.items():
            if getattr(dist, "is_atomic", False):
                # Constants contribute 0 inside support; -inf otherwise.
                # MCMC samplers hold these fixed, so we skip to keep
                # autodiff well-behaved.
                continue
            total = total + jnp.sum(dist.log_prob(params[name]))
        return total

    def log_density(self, params):
        """Target log-density for MCMC: log_prior - S_int.

        Falls back to log_prior when no interaction is attached.
        """
        lp = self.log_prior(params)
        if self._interaction is None:
            return lp
        S_int = self._interaction.action(
            self, params, method=self._action_method, **self._action_kwargs
        )
        return lp - S_int

    def sample_field(self, x_points, n_samples, key, sampler=None, batch_size=None):
        """Field values at multiple points, shape (n_samples, n_pts).

        Streams in chunks of `batch_size` parameter draws to bound memory;
        defaults to a single batch of `n_samples`.
        """
        sampler = sampler if sampler is not None else IIDSampler()
        x = jnp.atleast_2d(jnp.asarray(x_points, dtype=jnp.float32))
        bs = int(batch_size) if batch_size is not None else int(n_samples)
        out = []
        done = 0
        sub_keys = jr.split(key, max(1, (n_samples + bs - 1) // bs))
        for chunk_idx, sub_key in enumerate(sub_keys):
            b = min(bs, n_samples - done)
            if b <= 0:
                break
            params = sampler.sample(self, b, sub_key)
            eval_one = lambda p: self.evaluate(x, p)
            out.append(jax.vmap(eval_one)(params))
            done += b
        return jnp.concatenate(out, axis=0)

    # ---- correlators ----
    def correlator(
        self,
        x_points,
        n_samples,
        key,
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
            n_configs == 1. Streams field-samples in chunks of
            `batch_size` for memory.

        Connected mode (`connected=True`): cumulant via moments-to-cumulants
            formula; only `n_configs == 1` is supported here.
        """
        sampler = sampler if sampler is not None else IIDSampler()

        if connected:
            if n_configs != 1:
                raise ValueError("connected=True requires n_configs == 1")
            samples = self.sample_field(
                x_points, n_samples, key, sampler=sampler, batch_size=batch_size
            )
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

        x = jnp.atleast_2d(jnp.asarray(x_points))
        n_pts = x.shape[0]
        if n_pts % n_configs != 0:
            raise ValueError(
                f"len(x_points)={n_pts} must be a multiple of "
                f"n_configs={n_configs}"
            )
        n_corr = n_pts // n_configs
        bs = int(batch_size) if batch_size is not None else int(n_samples)

        sum_b = jnp.zeros(n_configs)
        sum_b_sq = jnp.zeros(n_configs)
        n_total = 0
        done = 0
        sub_keys = jr.split(key, max(1, (n_samples + bs - 1) // bs))
        for sub_key in sub_keys:
            b = min(bs, n_samples - done)
            if b <= 0:
                break
            params = sampler.sample(self, b, sub_key)
            phi = jax.vmap(lambda p: self.evaluate(x, p))(params)  # (b, n_pts)
            prod = jnp.prod(
                phi.reshape(b, n_configs, n_corr), axis=2
            )  # (b, n_configs)
            sum_b = sum_b + prod.sum(axis=0)
            sum_b_sq = sum_b_sq + (prod * prod).sum(axis=0)
            n_total += b
            done += b

        mean = sum_b / n_total
        var = (sum_b_sq - n_total * mean * mean) / (n_total - 1)
        stderr = jnp.sqrt(jnp.maximum(var, 0.0) / n_total)

        if n_configs == 1:
            return float(mean[0]), float(stderr[0])
        return np.asarray(mean), np.asarray(stderr)

    def correlator_reweighted(
        self,
        x_points,
        n_samples,
        key,
        interaction,
        n_configs=1,
        batch_size=None,
        action_method="trans_sym",
        action_kwargs=None,
    ):
        """Reweighted estimator: sample params from the i.i.d. prior, weight
        by exp(-S_int), report <phi(x_1)...phi(x_n) e^{-S_int}> / <e^{-S_int}>
        for each config block.

        Returns (means, errors) of shape (n_configs,) or scalars.
        Errors are the leading-order delta-method estimate ignoring the
        denominator's variance.
        """
        sampler = IIDSampler()
        action_kwargs = dict(action_kwargs) if action_kwargs else {}
        x = jnp.atleast_2d(jnp.asarray(x_points))
        n_pts = x.shape[0]
        if n_pts % n_configs != 0:
            raise ValueError(
                f"len(x_points)={n_pts} must be a multiple of "
                f"n_configs={n_configs}"
            )
        n_corr = n_pts // n_configs
        bs = int(batch_size) if batch_size is not None else int(n_samples)

        # Per-config running sums in log-space-safe form is overkill at
        # these N; use naive accumulators.
        wsum = 0.0
        w2sum = 0.0
        prod_sum = jnp.zeros(n_configs)
        prod_sq_sum = jnp.zeros(n_configs)
        n_total = 0
        done = 0
        sub_keys = jr.split(key, max(1, (n_samples + bs - 1) // bs))
        for sub_key in sub_keys:
            b = min(bs, n_samples - done)
            if b <= 0:
                break
            mc_key, sub_key2 = jr.split(sub_key)
            params = sampler.sample(self, b, sub_key2)
            # field values: (b, n_pts)
            phi = jax.vmap(lambda p: self.evaluate(x, p))(params)
            # action per draw: (b,)
            # use fresh MC quadrature key when method needs it
            kwargs = dict(action_kwargs)
            if action_method == "real_space_mc" and "x_quad" not in kwargs:
                kwargs["key"] = mc_key
            S = jax.vmap(
                lambda p: interaction.action(self, p, method=action_method, **kwargs)
            )(params)
            w = jnp.exp(-S)                                       # (b,)
            prod = jnp.prod(
                phi.reshape(b, n_configs, n_corr), axis=2
            )                                                     # (b, n_configs)
            wprod = w[:, None] * prod
            wsum = wsum + float(w.sum())
            w2sum = w2sum + float((w * w).sum())
            prod_sum = prod_sum + wprod.sum(axis=0)
            prod_sq_sum = prod_sq_sum + (wprod * wprod).sum(axis=0)
            n_total += b
            done += b

        means = prod_sum / wsum
        # delta-method estimate: Var(num/denom) ~ Var(num)/denom^2.
        var_num = (prod_sq_sum - n_total * (prod_sum / n_total) ** 2) / max(
            n_total - 1, 1
        )
        stderr = jnp.sqrt(jnp.maximum(var_num, 0.0) / n_total) / abs(wsum / n_total)
        if n_configs == 1:
            return float(means[0]), float(stderr[0])
        return np.asarray(means), np.asarray(stderr)


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
