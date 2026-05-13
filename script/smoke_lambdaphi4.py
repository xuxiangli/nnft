"""Compare LambdaPhi4 action methods on the same sampled parameters (JAX).

Prints the action value, relative error versus the dense ``trans_sym``
momentum-space sum (the JAX reference), and wall time for:

- ``real_space_mc``       (Monte Carlo quadrature)
- ``real_space_hermite``  (Gauss-Hermite quadrature)
- ``trans_sym``           (dense signed-pair momentum sum, reference)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax.random as jr

from nnft import CosNetFT, IIDSampler, LambdaPhi4, Theory


D = 3
M = 1.0
ALPHA = 0.0
LAMBDA_UV = 100.0
REGULATOR = "gaussian"
LAMBDA_PHI4 = 0.5
L_IR = 100.0

N_VALUES = (10, 30, 50)
M_X = int(1e5)
N_HERMITE = 8
SEED = 1


def relative_error(value, reference):
    denom = abs(reference) if reference != 0.0 else 1.0
    return abs(value - reference) / denom


def timed(call):
    t0 = time.perf_counter()
    value = call()
    # block to ensure JAX compute completes before timing
    value = float(value)
    return value, time.perf_counter() - t0


def build_case(N, seed):
    arch = CosNetFT(
        d_in=D, m=M, alpha=ALPHA, Lambda=LAMBDA_UV, regulator=REGULATOR
    )
    theory = Theory(arch, N=N, param_dists=arch.default_dists())
    key = jr.PRNGKey(seed)
    batch = IIDSampler().sample(theory, 1, key)
    params = {k: v[0] for k, v in batch.items()}
    interaction = LambdaPhi4(LAMBDA_PHI4, L_IR, "gaussian")
    return theory, params, interaction, key


def compare_for_N(N, seed):
    theory, params, interaction, key = build_case(N, seed)
    key_mc = jr.fold_in(key, 1)

    ref, t_ref = timed(
        lambda: interaction.action(theory, params, method="trans_sym")
    )
    mc, t_mc = timed(
        lambda: interaction.action(
            theory, params, method="real_space_mc",
            M_x=M_X, key=key_mc,
        )
    )
    her, t_her = timed(
        lambda: interaction.action(
            theory, params, method="real_space_hermite",
            n_hermite=N_HERMITE,
        )
    )
    return [
        (N, "trans_sym(dense)", ref, 0.0, t_ref),
        (N, f"real_space_mc(M_x={M_X})", mc, relative_error(mc, ref), t_mc),
        (N, f"real_space_hermite(n={N_HERMITE})", her, relative_error(her, ref), t_her),
    ]


def main():
    print(
        f"LambdaPhi4 action comparison (d={D}, m={M}, alpha={ALPHA}, "
        f"Lambda={LAMBDA_UV}, L={L_IR}, lambda={LAMBDA_PHI4})"
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
