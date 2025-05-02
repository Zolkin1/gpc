from typing import Tuple, Any
import time
import matplotlib.pyplot as plt
import pickle
import jax
import jax.numpy as jnp
from hydrax.alg_base import SamplingBasedController, Trajectory
from mujoco import mjx

from gpc.envs import TrainingEnv, SimulatorState


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

    def _single_compute(x: SimulatorState, ctrl: SamplingBasedController, params: Any, key: jax.Array) -> None:
        """Performs the computation for a single state."""
        x = x.replace(data=env.reset(x.data, key))

        params, rollouts = compute_value(x.data, ctrl, params)

        min_cost_idx = jnp.argmin(jnp.sum(rollouts.costs, axis=1))
        costs = jnp.sum(rollouts.costs, axis=1)
        J_star = costs[min_cost_idx]
        U_star = rollouts.controls[min_cost_idx]

        return J_star, U_star, x

    print("Starting data collection...")
    start_time = time.perf_counter()

    # RNG
    key = jax.random.PRNGKey(0)

    # Initial params
    params = ctrl.init_params()

    # Initial state
    x = env.init_state(key)

    keys = jax.random.split(key, num_compute)
    J_stars, U_stars, xs = jax.vmap(_single_compute, in_axes=(None, None, None, 0))(x, ctrl, params, keys)

    print(f"J_stars shape: {J_stars.shape}, U_stars shape: {U_stars.shape}")

    # Go through a grid in the statespace
    with open(filename, 'wb') as f:
        data_chunk = {'J_star': J_stars, 'U_star': U_stars, 'state': xs}
        pickle.dump(data_chunk, f)

    end_time = time.perf_counter()
    print("End of data collection.")
    print(f"Time taken: {end_time - start_time}s")


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

def extract_data(filename: str, env: TrainingEnv) -> Tuple[jax.Array, jax.Array]:
    """Extract the data from a file."""
    J_star = jnp.empty(0)
    obs = jnp.empty(0)
    with open(filename, 'rb') as f:
        while True:
            try:
                data_chunk = pickle.load(f)
                J_star = jnp.append(J_star, data_chunk['J_star'])
                state = data_chunk['state']
                for i in range(J_star.shape[0]):
                    data = jax.tree.map(lambda x: x[i], state.data)
                    obs = jnp.append(obs, env.get_obs(data))
            except EOFError:
                break

    obs = obs.reshape(J_star.shape[0], 3)

    print("Data extracted.")
    return J_star, obs

