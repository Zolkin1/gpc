import time
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from hydrax.alg_base import SamplingBasedController
from mujoco import mjx

from gpc.envs import TrainingEnv
from gpc.policy import Policy


def test_interactive(
    env: TrainingEnv,
    policy: Policy,
    ctrl: SamplingBasedController,
    mj_data: mujoco.MjData = None,
    inference_timestep: float = 0.1,
    warm_start_level: float = 1.0,
) -> None:
    """Test a GPC policy with an interactive simulation.

    Args:
        env: The environment, which defines the system to simulate.
        policy: The GPC policy to test.
        mj_data: The initial state for the simulation.
        inference_timestep: The timestep dt to use for flow matching inference.
        warm_start_level: The warm start level to use for the policy.
    """
    rng = jax.random.key(0)
    task = env.task

    # Set up the policy
    policy = policy.replace(dt=inference_timestep)
    policy.model.eval()
    jit_policy = jax.jit(
        partial(policy.apply, warm_start_level=warm_start_level)
    )

    # Set up the mujoco simultion
    mj_model = task.mj_model
    if mj_data is None:
        mj_data = mujoco.MjData(mj_model)

    # Initialize the knot sequence
    actions = jnp.zeros((ctrl.num_knots, task.model.nu))
    eval_time = jnp.zeros((1))

    # Set up an observation function
    mjx_data = mjx.make_data(task.model)

    @jax.jit
    def get_obs(mjx_data: mjx.Data) -> jax.Array:
        """Get an observation from the mujoco data."""
        mjx_data = mjx.forward(task.model, mjx_data)  # update sites & sensors
        return env.get_obs(mjx_data)

    params = ctrl.init_params()

    # Run the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            st = time.time()

            # Get an observation
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
            )
            obs = get_obs(mjx_data)

            # Update the action sequence
            inference_start = time.time()
            rng, policy_rng = jax.random.split(rng)
            actions = jit_policy(actions, obs, policy_rng)
            params = params.replace(mean=actions)
            U = ctrl.get_action(params, eval_time)
            mj_data.ctrl[:] = U

            inference_time = time.time() - inference_start
            obs_time = inference_start - st
            print(
                f"  Observation time: {obs_time:.5f}s "
                f" Inference time: {inference_time:.5f}s",
                end="\r",
            )

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            elapsed = time.time() - st
            if elapsed < mj_model.opt.timestep:
                time.sleep(mj_model.opt.timestep - elapsed)

    # Save what was last in the print buffer
    print("")
