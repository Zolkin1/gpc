import jax
import jax.numpy as jnp
from gpc.tasks.double_cart_pole_value import DoubleCartPoleValue
from mujoco import mjx

from gpc.envs import TrainingEnv


class DoubleCartPoleValueEnv(TrainingEnv):
    """Training environment for the double cart-pole swingup task."""

    def __init__(self, episode_length: int) -> None:
        """Set up the training environment."""
        super().__init__(task=DoubleCartPoleValue(), episode_length=episode_length)

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, theta_rng, pos_rng, vel_rng = jax.random.split(rng, 4)

        thetas = jax.random.uniform(theta_rng, (2), minval=-3.14, maxval=3.14)
        pos = jax.random.uniform(pos_rng, (), minval=-2.8, maxval=2.8)
        qvel = jax.random.uniform(vel_rng, (3,), minval=-10.0, maxval=10.0)
        qpos = jnp.array([pos, thetas[0], thetas[1]])

        return data.replace(qpos=qpos, qvel=qvel)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe the velocity and sin/cos of the angles."""
        p = data.qpos[0]
        theta1 = data.qpos[1]
        theta2 = data.qpos[2]
        q_obs = jnp.array(
            [
                p,
                jnp.cos(theta1),
                jnp.sin(theta1),
                jnp.cos(theta2),
                jnp.sin(theta2),
            ]
        )
        return jnp.concatenate([q_obs, data.qvel])

    @property
    def observation_size(self) -> int:
        """The size of the observation space (includes sin and cos)."""
        return 8
