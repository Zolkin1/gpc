import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from gpc.tasks.task_value import TaskValue


class WalkerValue(TaskValue):
    """A planar biped tasked with walking forward."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/walker/scene.xml"
        )
        super().__init__(mj_model, planning_horizon=4, sim_steps_per_control_step=15,  trace_sites=["torso_site"])

        # Get sensor ids
        self.torso_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_position"
        )
        self.torso_velocity_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_subtreelinvel"
        )
        self.torso_zaxis_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_zaxis"
        )

        # Set the target velocity (m/s) and height
        # TODO: make these parameters
        self.target_velocity = 1.5
        self.target_height = 1.2

    def _get_torso_height(self, state: mjx.Data) -> jax.Array:
        """Get the height of the torso above the ground."""
        sensor_adr = self.model.sensor_adr[self.torso_position_sensor]
        return state.sensordata[sensor_adr + 2]  # px, py, pz

    def _get_torso_velocity(self, state: mjx.Data) -> jax.Array:
        """Get the horizontal velocity of the torso."""
        sensor_adr = self.model.sensor_adr[self.torso_velocity_sensor]
        return state.sensordata[sensor_adr]

    def _get_torso_deviation_from_upright(self, state: mjx.Data) -> jax.Array:
        """Get the deviation of the torso from the upright position."""
        sensor_adr = self.model.sensor_adr[self.torso_zaxis_sensor]
        return state.sensordata[sensor_adr + 2] - 1.0

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        height_cost = jnp.square(
            self._get_torso_height(state) - self.target_height
        )
        orientation_cost = jnp.square(
            self._get_torso_deviation_from_upright(state)
        )
        velocity_cost = jnp.square(
            self._get_torso_velocity(state) - self.target_velocity
        )
        state_cost = 10.0 * height_cost + 3.0 * orientation_cost + 1.0 * velocity_cost
        control_cost = jnp.sum(jnp.square(control))
        return state_cost + 0.1 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T).
        The cost is given by the neural network value function."""
        # Make the observation match what the network expects
        pz = state.qpos[0]  # base coordinates are (z, x, theta)
        theta = state.qpos[2]
        base_pos_data = jnp.array([jnp.cos(theta), jnp.sin(theta), pz])
        obs = jnp.concatenate([base_pos_data, state.qpos[3:], state.qvel])

        val_cost = self.value_fcn.approximate(obs)

        return val_cost