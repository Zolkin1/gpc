import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from gpc.tasks.task_value import TaskValue

class PendulumValue(TaskValue):
    """An inverted pendulum swingup task."""

    def __init__(
        self, planning_horizon: int = 20, sim_steps_per_control_step: int = 5
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/pendulum/scene.xml"
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["tip"],
        )

    def _distance_to_upright(self, state: mjx.Data) -> jax.Array:
        """Get a measure of distance to the upright position."""
        theta = state.qpos[0] - jnp.pi
        theta_err = jnp.array([jnp.cos(theta) - 1, jnp.sin(theta)])
        return jnp.sum(jnp.square(theta_err))

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        theta_cost = self._distance_to_upright(state)
        theta_dot_cost = 0.01 * jnp.square(state.qvel[0])
        control_cost = 0.001 * jnp.sum(jnp.square(control))
        total_cost = theta_cost + theta_dot_cost + control_cost
        return total_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T).
        The cost is given by the neural network value function."""
        # Make the observation match what the network expects
        obs = jnp.stack([jnp.cos(state.qpos[0]), jnp.sin(state.qpos[0]), state.qvel[0]])

        val_cost = self.value_fcn.approximate(obs)

        return val_cost