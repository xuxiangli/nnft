"""Compare LambdaPhi4 action methods on the same sampled parameters.

Prints the action value, relative error versus the exact momentum-space sum,
and wall time for:

- explicit momentum-space summation
- real-space Monte Carlo integration
- approximate momentum-conservation pair summation
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from nnft import CosNetFT, IIDSampler, LambdaPhi4, Theory


D = 3
M = 1.0
ALPHA = 0.0
LAMBDA_UV = 100.0
REGULATOR = "gaussian"
LAMBDA_PHI4 = 0.5
L_IR = 100.0

N_VALUES = (10, 30, 50)
M_X = 1e6
MOMENTUM_EPS = 1e-5
SEED = 1

"""Output:
   N  method                                         S_int     rel.err    time [s]
------------------------------------------------------------------------------------
  10  explicit                                2.055077e+03   0.000e+00      0.0021
  10  real_space_mc(M_x=1000000.0)            2.056195e+03   5.440e-04      0.1413
  10  trans_sym(eps=1e-08)                    2.055077e+03   0.000e+00      0.0005
  30  explicit                                2.127185e+03   0.000e+00      0.1720
  30  real_space_mc(M_x=1000000.0)            2.125163e+03   9.506e-04      0.4058
  30  trans_sym(eps=1e-08)                    2.127185e+03   1.122e-13      0.0016
  50  explicit                                2.141606e+03   0.000e+00      1.3759
  50  real_space_mc(M_x=1000000.0)            2.151659e+03   4.694e-03      0.6743
  50  trans_sym(eps=1e-08)                    2.141606e+03   1.042e-12      0.0044
  80  explicit                                2.149768e+03   0.000e+00     10.2285
  80  real_space_mc(M_x=1000000.0)            2.156534e+03   3.147e-03      0.9918
  80  trans_sym(eps=1e-08)                    2.149768e+03   5.441e-12      0.0108
 100  explicit                                2.152853e+03   0.000e+00     31.6638
 100  real_space_mc(M_x=1000000.0)            2.157165e+03   2.003e-03      1.0879
 100  trans_sym(eps=1e-08)                    2.152853e+03   3.091e-12      0.0167
"""

def relative_error(value, reference):
    denom = abs(reference) if reference != 0.0 else 1.0
    return abs(value - reference) / denom


def timed(call):
    t0 = time.perf_counter()
    value = call()
    return value, time.perf_counter() - t0


def build_case(N, seed):
    arch = CosNetFT(
        d_in=D, m=M, alpha=ALPHA, Lambda=LAMBDA_UV, regulator=REGULATOR
    )
    theory = Theory(arch, N=N, param_dists=arch.default_dists())
    rng = np.random.default_rng(seed)
    params = IIDSampler().sample(theory, n_samples=1, rng=rng)
    interaction = LambdaPhi4(LAMBDA_PHI4, L_IR, "gaussian")
    return theory, params, interaction


def compare_for_N(N, seed):
    theory, params, interaction = build_case(N, seed)

    exact_arr, exact_s = timed(
        lambda: interaction.action(theory, params, b=1, method="explicit")
    )
    exact = float(exact_arr[0])

    mc_arr, mc_s = timed(
        lambda: interaction.action(
            theory,
            params,
            b=1,
            method="real_space_mc",
            M_x=M_X,
            rng=np.random.default_rng(seed + 10_000),
        )
    )
    mc = float(mc_arr[0])

    mom_arr, mom_s = timed(
        lambda: interaction.action(
            theory,
            params,
            b=1,
            method="trans_sym",
            eps=MOMENTUM_EPS,
        )
    )
    momentum = float(mom_arr[0])

    rows = [
        (N, "explicit", exact, 0.0, exact_s),
        (N, f"real_space_mc(M_x={M_X})", mc, relative_error(mc, exact), mc_s),
        (
            N,
            f"trans_sym(eps={MOMENTUM_EPS:g})",
            momentum,
            relative_error(momentum, exact),
            mom_s,
        ),
    ]
    return rows


def main():
    print(
        "LambdaPhi4 action comparison "
        f"(d={D}, m={M}, alpha={ALPHA}, Lambda={LAMBDA_UV}, "
        f"L={L_IR}, lambda={LAMBDA_PHI4})"
    )
    print()
    print(f"{'N':>4}  {'method':<36}  {'S_int':>14}  {'rel.err':>10}  {'time [s]':>10}")
    print("-" * 84)
    for idx, N in enumerate(N_VALUES):
        for row in compare_for_N(N, SEED + idx):
            n, method, value, rel, seconds = row
            print(f"{n:4d}  {method:<36}  {value:14.6e}  {rel:10.3e}  {seconds:10.4f}")


if __name__ == "__main__":
    main()
