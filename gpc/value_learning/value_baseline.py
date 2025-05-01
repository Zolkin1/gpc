from typing import Tuple, Any
import time
import matplotlib.pyplot as plt
import pickle
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
    optimize_jit = jax.jit(ctrl.optimize)
    params, rollouts = optimize_jit(x, params)

    return params, rollouts

def compute_baseline(filename: str, ctrl: SamplingBasedController, env: TrainingEnv, num_compute: int) -> jax.Array:
    """Compute the value function baseline for a given controller."""

    start_time = time.perf_counter()

    # RNG
    key = jax.random.PRNGKey(0)

    # Initial params
    params = ctrl.init_params()

    # Initial state
    x = env.init_state(key)

    # Go through a grid in the statespace
    with open(filename, 'wb') as f:
        for i in range(num_compute):
            key = jax.random.PRNGKey(i)
            # Get a state
            x = x.replace(data=env.reset(x.data, key))
            # q = 1.5*jnp.ones(1)
            # x = x.replace(data=x.data.replace(qpos=q))

            # Compute the value at each point
            params, rollouts = compute_value(x.data, ctrl, params)

            # Single optimization result
            min_cost_idx = jnp.argmin(jnp.sum(rollouts.costs, axis=1))
            costs = jnp.sum(rollouts.costs, axis=1)
            J_star = costs[min_cost_idx]
            U_star = rollouts.controls[min_cost_idx]
            print(f"[OL] average rollout cost: {jnp.mean(costs)}, standard dev rollout cost: {jnp.std(costs)}"
                  f" minimum rollout cost: {J_star}")

            data_chunk = {'J_star': J_star, 'U_star': U_star, 'state': x}
            pickle.dump(data_chunk, f)

    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time}s")

    # x_traj = rollouts.trace_sites[min_cost_idx, :, 0, :]

    # time = jnp.linspace(params.tk[0], params.tk[-1], U_star.shape[0])
    # time_x = jnp.linspace(params.tk[0], params.tk[-1], x_traj.shape[0])

    # fig, axes = plt.subplots(4, 1)
    # axes[0].plot(time, U_star)
    # axes[0].set_ylabel("U")
    # for i in range(x_traj.shape[1]):
    #     axes[i + 1].plot(time_x, x_traj[:, i])
    #     label = ["X", "Y", "Z"]
    #     axes[i + 1].set_ylabel(label[i])
    # plt.show()

    # Save to a file
    # csv saved as (state, value, trajectory)

def parse_value_data(filename: str) -> None:
    """Parse the value data from a file."""
    J_star = jnp.empty(0)
    U_star = jnp.empty(0)
    x = jnp.empty(0)
    y = jnp.empty(0)
    with open(filename, 'rb') as f:
        while True:
            try:
                data_chunk = pickle.load(f)
                J_star = jnp.append(J_star, data_chunk['J_star'])
                U_star = jnp.append(U_star, data_chunk['U_star'])
                # state = jnp.append(state, data_chunk['state'])
                state = data_chunk['state']
                x = jnp.append(x, state.data.qpos[0])
                y = jnp.append(y, state.data.qvel[0])
                print("Loading data...")
            except EOFError:
                break

    plt.figure()
    plt.scatter(x, y, c=J_star)
    plt.colorbar()
    plt.show()