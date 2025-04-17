import jax
import jax.numpy as jnp
from gpc.tasks.walker_value import WalkerValue
from hydrax.tasks.walker import Walker
from mujoco import mjx

from gpc.envs import TrainingEnv


class WalkerValueEnv(TrainingEnv):
    """Training environment for the walker task."""

    def __init__(self, episode_length: int) -> None:
        """Set up the walker training environment."""
        super().__init__(task=WalkerValue(), episode_length=episode_length)

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)

        # Joint limits are zero for the floating base
        q_min = self.task.model.jnt_range[:, 0]
        q_max = self.task.model.jnt_range[:, 1]
        q_min = q_min.at[2].set(-1.5)  # orientation
        q_max = q_max.at[2].set(1.5)
        qpos = jax.random.uniform(pos_rng, (9,), minval=q_min, maxval=q_max)
        qvel = jax.random.uniform(vel_rng, (9,), minval=-0.1, maxval=0.1)

        return data.replace(qpos=qpos, qvel=qvel)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe everything in the state except the horizontal position."""
        pz = data.qpos[0]  # base coordinates are (z, x, theta)
        theta = data.qpos[2]
        base_pos_data = jnp.array([jnp.cos(theta), jnp.sin(theta), pz])
        return jnp.concatenate([base_pos_data, data.qpos[3:], data.qvel])

    @property
    def observation_size(self) -> int:
        """The size of the observation space."""
        return 18
