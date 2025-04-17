import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT

from gpc.tasks.task_value import TaskValue


class DoubleCartPoleValue(TaskValue):
    """A swing-up task for a double pendulum on a cart."""

    def __init__(
        self, planning_horizon: int = 10, sim_steps_per_control_step: int = 8
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/double_cart_pole/scene.xml"
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["tip"],
        )

        self.tip_id = mj_model.site("tip").id

    def _distance_to_upright(self, state: mjx.Data) -> jax.Array:
        """Get a measure of distance to the upright position."""
        tip_z = state.site_xpos[self.tip_id, 2]
        tip_x = state.site_xpos[self.tip_id, 0]
        cart_x = state.qpos[0]
        return jnp.square(tip_z - 4.0) + jnp.square(tip_x - cart_x)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        upright_cost = self._distance_to_upright(state)
        velocity_cost = 0.1 * jnp.sum(jnp.square(state.qvel[1:]))
        control_cost = 0.001 * jnp.sum(jnp.square(control))
        return upright_cost + velocity_cost + control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        p = state.qpos[0]
        theta1 = state.qpos[1]
        theta2 = state.qpos[2]
        q_obs = jnp.array(
            [
                p,
                jnp.cos(theta1),
                jnp.sin(theta1),
                jnp.cos(theta2),
                jnp.sin(theta2),
            ]
        )
        obs = jnp.concatenate([q_obs, state.qvel])

        val_cost = self.value_fcn.approximate(obs)

        return val_cost
