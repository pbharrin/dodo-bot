"""Start a new training stage from a previous stage's final policy.

Brax's `--restore` reloads a checkpoint as-is. That breaks when the new stage
feeds inputs the old one never varied. Stage 1 always had a zero velocity
command, so its observation normalizer stores std = 1e-6 for the three command
inputs. A 0.3 m/s command would then reach the network as ~3e5, and because the
normalizer averages over all past steps (150M) it would take a very long time to
adapt.

`warm_start_params` fixes that:
  * Command inputs: normalizer mean/std are set to the new command
    distribution (estimated by sampling the new env's commands).
  * First-layer weights on the command inputs (policy and value) are zeroed.
    Those weights never received gradient in stage 1 (the input was always 0),
    so they are still random init; zeroing them means the warm-started policy
    initially behaves exactly like the stage-1 policy, whatever the command, and
    learns to use the command from there.
Only command inputs that never varied in the source run (normalizer std below
1e-3) are adapted. Warm-starting from a run that already saw commands (e.g.
stage 2 -> stage 2 with a new reward) keeps them, and the policy, unchanged.
Everything else (stepping, balance, push recovery, value function) is kept.
"""
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.io import model as brax_model
from brax.training import types as training_types

COMMAND = slice(6, 9)          # command position in both 'state' and 'privileged_state'
_MIN_STD = 1e-3                # below this the input is treated as constant and left alone


def _count(c) -> float:
    if isinstance(c, training_types.UInt64):
        return float(c.hi) * 2.0 ** 32 + float(c.lo)
    return float(c)


def command_statistics(env, num_samples: int = 200_000, seed: int = 0):
    """Mean/std of the command input over the new env's episodes (stand + step mix)."""
    cfg = env._config
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    stepping = (jax.random.uniform(k1, (num_samples,)) > cfg.stand_probability).astype(jp.float32)
    cmd = jax.vmap(env.sample_command)(jax.random.split(k2, num_samples), stepping)
    cmd = np.asarray(cmd)
    return cmd.mean(0), cmd.std(0)


def warm_start_params(params, env):
    """params = (normalizer, policy, value) from a previous run's params.pkl."""
    normalizer, policy, value = params
    mean_c, std_c = command_statistics(env)
    src_std = np.asarray(normalizer.std['state'])[COMMAND]
    varied = (std_c > _MIN_STD) & (src_std < _MIN_STD)   # newly varying inputs only
    count = _count(normalizer.count)

    def fix_stats(key):
        mean = np.array(normalizer.mean[key])
        std = np.array(normalizer.std[key])
        sv = np.array(normalizer.summed_variance[key])
        idx = np.arange(mean.shape[0])[COMMAND][varied]
        mean[idx] = mean_c[varied]
        std[idx] = std_c[varied]
        sv[idx] = std_c[varied] ** 2 * count          # Welford: std = sqrt(sv / count)
        return jp.asarray(mean), jp.asarray(std), jp.asarray(sv)

    keys = list(normalizer.mean.keys())
    fixed = {k: fix_stats(k) for k in keys}
    normalizer = normalizer.replace(
        mean={k: fixed[k][0] for k in keys},
        std={k: fixed[k][1] for k in keys},
        summed_variance={k: fixed[k][2] for k in keys},
    )

    def zero_command_rows(net):
        net = jax.tree_util.tree_map(lambda x: x, net)       # copy
        kernel = np.array(net['params']['hidden_0']['kernel'])
        kernel[np.arange(kernel.shape[0])[COMMAND][varied]] = 0.0
        net['params']['hidden_0']['kernel'] = jp.asarray(kernel)
        return net

    return normalizer, zero_command_rows(policy), zero_command_rows(value), {
        'command_mean': mean_c.tolist(), 'command_std': std_c.tolist(),
        'adapted_command_inputs': varied.tolist(),
        'normalizer_count': count}


def load(run_dir: Path, env):
    params = brax_model.load_params(str(Path(run_dir) / 'params.pkl'))
    if len(params) != 3:
        raise ValueError(f'{run_dir}/params.pkl has {len(params)} entries, expected '
                         '(normalizer, policy, value)')
    normalizer, policy, value, info = warm_start_params(params, env)
    return (normalizer, policy, value), info
