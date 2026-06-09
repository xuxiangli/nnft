"""Autocorrelation diagnostics for MCMC observable time series.

These operate on a caller-supplied 1-D time series ``x`` (e.g. a per-sample
observable extracted from a sampler chain). The samplers in :mod:`nnft.samplers`
do not retain chains themselves, so the caller is responsible for collecting the
ordered series before calling these functions.

Conventions (kept identical to the original inline implementation in
``script/mh_diagnostic.py`` so existing results remain reproducible):

* ``rho(k)``    : normalized autocorrelation, ``rho(0) = 1``.
* ``tau_int``   : integrated autocorrelation time, ``1/2 + sum_{k>=1} rho(k)``,
                  truncated with a Sokal window.
* ``ESS``       : effective sample size, ``n / max(2 * tau_int, 1)``.
"""

import numpy as np

__all__ = [
    "autocorr_function",
    "integrated_autocorr",
    "effective_sample_size",
    "autocorr_summary",
]


def autocorr_function(x, n_lag=None):
    """Normalized autocorrelation rho(k) of a 1D series, k=0..n_lag-1.

    Uses an FFT-based (unbiased-lag-normalized) estimate of the autocovariance.
    ``rho(0)`` is ``1`` for any non-constant series; a constant (zero-variance)
    series returns the raw, un-normalized autocovariance (all zeros) rather than
    dividing by zero.

    Parameters
    ----------
    x : array_like
        1-D real time series.
    n_lag : int, optional
        Number of lags to return. Defaults to ``len(x) // 4``.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n_lag is None:
        n_lag = n // 4
    # FFT-based autocovariance
    f = np.fft.fft(x, n=2 * n)
    acf = np.real(np.fft.ifft(f * np.conjugate(f)))[:n_lag]
    acf /= (n - np.arange(n_lag))
    if acf[0] == 0:
        return acf
    return acf / acf[0]


def integrated_autocorr(x, c=5.0):
    """Sokal-windowed integrated autocorrelation time.

    ``tau_int(W) = 1/2 + sum_{k=1}^W rho(k)``; window W chosen as the smallest
    W >= c * tau_int(W). Returns ``(tau_int, W_chosen)``.

    Parameters
    ----------
    x : array_like
        1-D real time series.
    c : float, optional
        Sokal window factor (default 5.0). Larger ``c`` reduces bias at the cost
        of higher variance.
    """
    rho = autocorr_function(x, n_lag=min(len(x) // 2, 5000))
    tau_running = 0.5 + np.cumsum(rho[1:])
    W_arr = np.arange(1, len(tau_running) + 1)
    ok = W_arr >= c * tau_running
    if not np.any(ok):
        return float(tau_running[-1]), int(W_arr[-1])
    W = int(W_arr[np.argmax(ok)])
    return float(tau_running[W - 1]), W


def effective_sample_size(x, c=5.0):
    """Effective sample size ``n / max(2 * tau_int, 1)``.

    Parameters
    ----------
    x : array_like
        1-D real time series.
    c : float, optional
        Sokal window factor passed to :func:`integrated_autocorr`.
    """
    x = np.asarray(x, dtype=float)
    tau_int, _ = integrated_autocorr(x, c=c)
    return len(x) / max(2.0 * tau_int, 1.0)


def autocorr_summary(x, c=5.0):
    """Convenience summary of an observable series.

    Returns a dict with the mean, sample spread, naive and ESS-corrected
    standard errors, integrated autocorrelation time, Sokal window, and ESS::

        {mean, std, naive_se, ess_se, tau_int, window, ess, n}

    where ``naive_se = std(ddof=1) / sqrt(n)`` and
    ``ess_se = naive_se * sqrt(max(2 * tau_int, 1))``.

    Parameters
    ----------
    x : array_like
        1-D real time series.
    c : float, optional
        Sokal window factor passed to :func:`integrated_autocorr`.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    naive_se = std / np.sqrt(n)
    tau_int, W = integrated_autocorr(x, c=c)
    ess_se = naive_se * np.sqrt(max(2.0 * tau_int, 1.0))
    ess = n / max(2.0 * tau_int, 1.0)
    return {
        "mean": mean,
        "std": std,
        "naive_se": float(naive_se),
        "ess_se": float(ess_se),
        "tau_int": float(tau_int),
        "window": int(W),
        "ess": float(ess),
        "n": int(n),
    }
