import argparse

import mujoco
from flax import nnx
from hydrax.algs import PredictiveSampling, MPPI
from hydrax.simulation.deterministic import run_interactive as run_sampling
from hydrax.tasks.pendulum import Pendulum

from gpc.architectures import DenoisingMLP
from gpc.envs import PendulumValueEnv
from gpc.policy import Policy
from gpc.sampling import BootstrappedPredictiveSampling
from gpc.testing import test_interactive
from gpc.training import train
from gpc.value_learning.value_verification import verify_value_fcn
from gpc.value_training import Fvi


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Swing up an inverted pendulum"
    )
    subparsers = parser.add_subparsers(
        dest="task", help="What to do (choose one)"
    )
    subparsers.add_parser("train", help="Train (and save) a generative policy")
    subparsers.add_parser("test", help="Test a generative policy")
    subparsers.add_parser(
        "sample", help="Bootstrap sampling-based MPC with a generative policy"
    )
    subparsers.add_parser("value_verification", help="Verify the value function.")
    args = parser.parse_args()

    # Set up the environment and save file
    env = PendulumValueEnv(episode_length=200)
    save_file = "/tmp/pendulum_policy.pkl"

    value_save_file = "/tmp/pendulum_value_fn.pkl"

    if args.task == "train":
        # Train the policy and save it to a file
        ctrl = PredictiveSampling(env.task, num_samples=8, noise_level=0.1)
        net = DenoisingMLP(
            action_size=env.task.model.nu,
            observation_size=env.observation_size,
            horizon=env.task.planning_horizon,
            hidden_layers=[32, 32],
            rngs=nnx.Rngs(0),
        )
        policy, value_fcn = train(
            env,
            ctrl,
            net,
            num_policy_samples=2,
            log_dir="/tmp/gpc_pendulum",
            num_epochs=10,
            num_iters=10,
            num_envs=128,
            num_videos=2,
            strategy="policy",
            with_value=True,
        )
        policy.save(save_file)
        value_fcn.save(value_save_file)
        print(f"Saved policy to {save_file}")
        print(f"Saved value function to {value_save_file}")

    elif args.task == "test":
        # Load the policy from a file and test it interactively
        print(f"Loading policy from {save_file}")
        policy = Policy.load(save_file)
        test_interactive(env, policy)

    elif args.task == "sample":
        # Use the policy to bootstrap sampling-based MPC
        policy = Policy.load(save_file)
        ctrl = BootstrappedPredictiveSampling(
            policy,
            env.get_obs,
            num_policy_samples=4,
            task=env.task,
            num_samples=4,
            noise_level=0.1,
        )

        mj_model = env.task.mj_model
        mj_data = mujoco.MjData(mj_model)
        run_sampling(ctrl, mj_model, mj_data, frequency=50)

    elif args.task == "value_verification":
        print("Verifying the value function...")
        print(f"Loading policy from {value_save_file}")
        # Load the value function network
        value_fcn = Fvi.load(value_save_file)

        # Make the tasks and controller
        task = Pendulum(planning_horizon=60)
        env.task = task
        # ctrl = MPPI(env.task, 32, 0.1, 10)
        ctrl = PredictiveSampling(env.task, num_samples=32, noise_level=0.1)


        # Call the verification function
        verify_value_fcn(env=env,
                         ctrl=ctrl,
                         nstates=5,
                         mpc_iterations=10,
                         value_iterations=4,
                         state_size=2
                         )

    else:
        parser.print_help()
