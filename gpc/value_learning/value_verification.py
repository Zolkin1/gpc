import time
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple, Union
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController

from gpc.augmented import PACParams, PolicyAugmentedController
from gpc.envs import SimulatorState, TrainingEnv
from gpc.tasks.task_value import TaskValue, Task
from gpc.value_training import Fvi

def verify_value_fcn(
        env: TrainingEnv,
        ctrl: SamplingBasedController,
        nstates: int,
        mpc_iterations: int,
        value_iterations: int,
        state_size: int
):
    """Verifies that the learned value function is close to approximate ground truth values.

    To check the value function, a large number of states will be checked.

    A "ground truth" value estimator will be computed for each state. The ground truth value will be approximated with a
    long horizon, high-sample, and, ideally, decreasing noise, MPPI computation.

    To be clear, the states will need to be distilled down into observations to evaluate the value function approximation.

    For each state, this "ground truth" estimate will be computed multiple times to be sure the estimate is trusted.

    Then, the ground truth estimate can be averaged over all the state to get a single number to describe the value and
    be compared to the estimated value. Of course this single number looses a lot of information, so individual state
    comparisons should also be done.

    There will also be an option to train a value function based on this data. The ground truth data should also be
    stored somewhere so that it can be re-used.

    """

    if nstates <= 0:
        raise ValueError("nstates must be positive!")

    rng = jax.random.PRNGKey(0)

    s0 = jnp.zeros((nstates, state_size))

    for i in range(nstates):
        # Make new rng
        rng, env_rng = jax.random.split(rng)

        # Get an initial state to verify
        x0 = env.init_state(env_rng)
        s0 = s0.at[i, :].set(jnp.concatenate([x0.data.qpos, x0.data.qvel]))

        # Compute the value function estimate at this location
        mean, stdev, min_cost = compute_value(x0, ctrl, mpc_iterations, value_iterations)

        print(f"mean: {mean}, stdev: {stdev}, min_cost: {min_cost}")

    print(f"s0: {s0}")

    # TODO: Make a plot of the state vs the value function estimate

    # TODO: Make a plot of the state vs the value function approximation

    # TODO: Quantify the difference in value function esimtation

def compute_value(
        state: SimulatorState,
        ctrl: SamplingBasedController,
        mpc_iterations: int,
        value_iterations: int,
) -> Tuple[jax.Array, jax.Array]:
    """Compute the value of the value function and std. dev. at the given state."""

    def _mpc_iters(psi: Any, i: int) -> Tuple[jax.Array, jax.Array]:
        # TODO: Try scaling the noise level with every iteration here
        psi, rollouts = ctrl.optimize(state.data, psi)
        costs = jnp.sum(rollouts.costs, axis=1)
        # jax.debug.print("rollout costs: {}", costs)

        min_cost = jnp.min(costs)

        # jax.debug.print("min cost: {}", min_cost)

        return psi, min_cost

    def _value_iters(costs: jax.Array, i: int) -> Tuple[jax.Array, int]:
        # Perform a number of MPC iterations
        miter = jnp.arange(mpc_iterations)
        psi = ctrl.init_params()
        psi = psi.replace(rng=jax.random.PRNGKey(i))        # TODO: Consider using a better seed here
        _, min_costs = jax.lax.scan(_mpc_iters, psi, miter)

        costs = costs.at[i].set(min_costs[-1])

        # jax.debug.print("{}", costs)

        return costs, i

    viter = jnp.arange(value_iterations)
    costs = jnp.zeros(value_iterations)
    costs, _ = jax.lax.scan(_value_iters, costs, viter)

    # Compute the mean cost and the standard deviation
    mean_cost = jnp.mean(costs)
    std_cost = jnp.std(costs)
    min_cost = jnp.min(costs)

    return mean_cost, std_cost, min_cost



