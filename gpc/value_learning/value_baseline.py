from typing import Tuple, Any
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from hydrax.alg_base import SamplingBasedController, Trajectory
from mujoco import mjx

from gpc.envs import TrainingEnv


# Run a very fine traj opt with lots of samples and iterations.
# The idea is that the trajopt will find the optimal solution so that the resulting cost can be used as the value function.

def compute_value(x: mjx.Data, ctrl: SamplingBasedController, params: Any) -> (
        Tuple)[Any, Trajectory,]:
    """Given a state, compute the value function value and the associated optimal trajectory."""
    params_1, rollouts_1 = ctrl.optimize(x, params)

    # # Closed loop rollout
    # def _scan_fn(carry, _):
    #     x, params = carry
    #     params, rollout = ctrl.optimize(x, params)
    #     min_cost_idx = jnp.argmin(jnp.sum(rollout.costs, axis=1))
    #     x = x.replace(ctrl=rollout.controls[min_cost_idx, 0])
    #     x_new = mjx.step(ctrl.model, x)
    #     return (x_new, params), (rollout.controls[min_cost_idx, 0], rollout.costs[min_cost_idx, 0],
    #                          rollout.trace_sites[min_cost_idx, 0])
    #
    # (xf, params), (U_star, J_star, traces) = jax.lax.scan(_scan_fn, (x, params),
    #                                                       jnp.arange(ctrl.ctrl_steps))

    # print(f"J_star shape: {J_star.shape}")
    # print(f"U_star shape: {U_star.shape}")
    # print(f"traces shape: {traces.shape}")

    return params, rollouts_1

def compute_baseline(ctrl: SamplingBasedController, env: TrainingEnv) -> jax.Array:
    """Compute the value function baseline for a given controller."""

    # RNG
    key = jax.random.PRNGKey(0)

    # Initial params
    params = ctrl.init_params()

    # Go through a grid in the statespace


    # Get a state
    x = env.init_state(key)
    q = 1.5*jnp.ones(1)
    x = x.replace(data=x.data.replace(qpos=q))

    # Compute the value at each point
    params, rollouts = compute_value(x.data, ctrl, params)

    # Single optimization result
    min_cost_idx = jnp.argmin(jnp.sum(rollouts.costs, axis=1))
    costs = jnp.sum(rollouts.costs, axis=1)
    J_star = costs[min_cost_idx]
    U_star = rollouts.controls[min_cost_idx]
    print(f"[OL] average rollout cost: {jnp.mean(costs)}, standard dev rollout cost: {jnp.std(costs)}"
          f" minimum rollout cost: {J_star}")

    print(f"trace site shape: {rollouts.trace_sites.shape}")
    x_traj = rollouts.trace_sites[min_cost_idx, :, 0, :]

    time = jnp.linspace(params.tk[0], params.tk[-1], U_star.shape[0])
    time_x = jnp.linspace(params.tk[0], params.tk[-1], x_traj.shape[0])

    fig, axes = plt.subplots(4, 1)
    axes[0].plot(time, U_star)
    axes[0].set_ylabel("U")
    for i in range(x_traj.shape[1]):
        axes[i + 1].plot(time_x, x_traj[:, i])
        label = ["X", "Y", "Z"]
        axes[i + 1].set_ylabel(label[i])
    plt.show()

    # Save to a file
    # csv saved as (state, value, trajectory)