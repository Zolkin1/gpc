from typing import Tuple, Any

import jax.lax
import jax.numpy as jnp

from hydrax.alg_base import SamplingBasedController

from gpc.envs import TrainingEnv, SimulatorState


def simulate_episode(
        env: TrainingEnv,
        ctrl: SamplingBasedController,
        rng: jax.Array,
):
    """Start from a random state and simulate a closed loop rollout."""
    def _sim_step(carry: Tuple[Any, SimulatorState], t_sim: int
                  ) -> Tuple[Tuple[Any, SimulatorState], Tuple[jax.Array, jax.Array, jax.Array,]]:
        psi, x = carry

        # Optimize the controller
        psi, rollouts = ctrl.optimize(x.data, psi)

        # Get the control input
        t = psi.tk[0]
        U_star = ctrl.get_action(psi, t)

        # Apply the control to the sim
        x = env.step(x, U_star)

        # Get the optimal cost
        # TODO: Technically would need to rollout the optimal actions
        costs = jnp.sum(rollouts.costs, axis=1)
        min_cost_idx = jnp.argmin(costs)

        return (psi, x), (U_star, costs[min_cost_idx], env.get_obs(x.data))

    rng, env_rng = jax.random.split(rng)

    params = ctrl.init_params()
    x = env.init_state(env_rng)
    _, (inputs, costs, obs) = jax.lax.scan(f=_sim_step, init=(params, x),
                                           xs=jnp.arange(env.episode_length))

    return inputs, costs, obs


def gather_data(env: TrainingEnv, ctrl: SamplingBasedController, num_envs: int, rng: jax.Array,):
    """Gather data from all the environments and all the episodes."""

    @jax.jit
    def _simulate_episode_envs(rng: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Simulate multiple episodes in parallel."""

        rng_env = jax.random.split(rng, num_envs)

        # For now, just sim one env
        inputs, costs, obs = jax.vmap(simulate_episode, in_axes=(None, None, 0))(env, ctrl, rng_env)

        return inputs, costs, obs

    inputs, costs, obs = _simulate_episode_envs(rng)

    print("Data gathered.")
    print(f"inputs shape: {inputs.shape}, costs shape: {costs.shape}")

    # Re-format the data for the value training
    # Flatten over the initial conditions and envs
    J_star = costs.reshape(-1, 1)
    obs = obs.reshape(-1, env.observation_size)

    print(f"J_star shape: {J_star.shape}, obs shape: {obs.shape}")

    return J_star, obs
