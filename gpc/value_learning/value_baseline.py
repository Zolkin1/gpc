from pyexpat import model
from typing import Tuple, Any
import time
import matplotlib.pyplot as plt
import numpy as np
import optax
import pickle
import jax
import jax.numpy as jnp
from flax import nnx
import os
from hydrax.alg_base import SamplingBasedController, Trajectory
from mujoco import mjx
from tensorboardX import SummaryWriter

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
                obs = jnp.append(obs, env.get_obs(state.data))
            except EOFError:
                break

    obs = obs.reshape(J_star.shape[0], 3)

    print("Data extracted.")
    return J_star, obs


@nnx.jit
def fit_value_function(model: nnx.Module,
                       optimizer: nnx.Optimizer,
                       J_star: jax.Array,
                       obs: jax.Array,
                       # batch_size: int,
                       # num_epochs: int,
                       rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Fit a value function to the data."""
    batch_size = 1 #5 #128
    num_epochs = 200
    num_data_points = J_star.shape[0]
    num_batches = max(1, num_data_points // batch_size)

    def _loss_fn(model: nnx.Module, obs: jax.Array, value_targets: jax.Array) -> jax.Array:
        """Fitted Value Iteration loss function."""
        pred = model(obs)
        return 0.5 * jnp.mean(jnp.square(pred - value_targets))

    def _train_step(model: nnx.Module, optimizer: nnx.Optimizer, rng: jax.Array) -> Tuple:
        """Take a single gradient descent step on a batch of data."""
        rng, batch_rng = jax.random.split(rng)
        batch_idx = jax.random.randint(batch_rng, (batch_size,), 0, num_data_points)
        batch_obs = obs[batch_idx]
        batch_value_targets = J_star[batch_idx]

        loss, grad = nnx.value_and_grad(_loss_fn)(model, batch_obs, batch_value_targets)

        optimizer.update(grad)

        grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grad)]))

        return rng, loss, grad_norm

    # Epoch loop (through all data)
        # Batch loop (through all the batches)
    @nnx.scan
    def _scan_fn(carry: Tuple, _: int) -> Tuple:
        """Scan function for the gradient descent."""
        rng, optimizer, model = carry
        rng, loss, grad_norm = _train_step(model, optimizer, rng)
        return (rng, optimizer, model), (loss, grad_norm)

    print(f"Total train steps: {num_epochs*num_batches}")
    _, (losses, grad_norms) = _scan_fn((rng, optimizer, model), jnp.arange(num_batches*num_epochs))

    jax.debug.print("grad norms: {}", grad_norms)

    return losses, grad_norms

def value_train(env: TrainingEnv,
        net: nnx.Module,
        filename: str,
        learning_rate: float = 1e-3,
      )->None:
    """Train a value function approximator."""
    # Print some info about the policy architecture
    params = nnx.state(net, nnx.Param)
    total_params = sum([np.prod(x.shape) for x in jax.tree.leaves(params)], 0)
    print(f"Policy: {type(net).__name__} with {total_params} parameters")
    print("")

    # Set up the optimizer
    optimizer = nnx.Optimizer(net, optax.adamw(learning_rate))

    # Set up the policy
    normalizer = nnx.BatchNorm(
        num_features=env.observation_size,
        momentum=0.1,
        use_bias=False,
        use_scale=False,
        use_fast_variance=False,
        rngs=nnx.Rngs(0),
    )

    J_star, obs = extract_data(filename, env)
    print(f"obs size: {obs.shape}")
    print(f"{J_star.shape[0]} data points.")

    losses, grad_norms = fit_value_function(net, optimizer, J_star, obs, jax.random.PRNGKey(0))
    print(f"Loss: {losses[-1]}")

    # Define log directory
    log_dir = os.path.join("logs", "value_function_fitting", str(int(time.time())))
    # Create the directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a SummaryWriter
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be written to: {log_dir}")

    for i in range(losses.shape[0]):
        writer.add_scalar("grad_norm", grad_norms[i], i)
        writer.add_scalar("loss", losses[i], i)

    writer.close()

    if obs.shape[1] == 3:   # Just for the pendulum
        # Plot the value network to see how it looks
        n_points = 100
        theta_net = jnp.linspace(-np.pi, np.pi, n_points)
        thetadot_net = jnp.linspace(-10, 10, n_points)

        X, Y = jnp.meshgrid(theta_net, thetadot_net)

        # Now make this into observation points
        st = jnp.sin(X)
        ct = jnp.cos(X)

        obs_points = jnp.stack([ct.reshape(-1), st.reshape(-1), Y.reshape(-1)], axis=-1)
        # print(f"obs size: {obs_points.shape}")
        # print(f"st size: {st.shape}")
        # print(f"ct size: {ct.shape}")
        # print(f"Y size: {Y.shape}")

        Z_flat = net(obs_points)
        # print(f"Z_flat size: {Z_flat.shape}")
        Z = Z_flat.reshape((n_points,n_points))
        # print(f"Z size: {Z.shape}")

        plt.imshow(Z, origin='lower', aspect='auto', cmap='viridis', extent=(X.min(), X.max(), Y.min(), Y.max()))
        theta = jnp.arctan2(obs[:, 1], obs[:, 0]).reshape(-1)
        theta_dot = obs[:, 2]
        plt.scatter(theta, theta_dot, c=J_star, cmap='viridis')
        plt.colorbar(label='Z Value')
        plt.xlabel('theta')
        plt.ylabel('theta dot')
        plt.title('Fit Value Function')
        plt.show()

        plt.figure()
        plt.imshow(Z, origin='lower', aspect='auto', cmap='viridis', extent=(X.min(), X.max(), Y.min(), Y.max()))
        plt.colorbar(label='Z Value')
        plt.xlabel('theta')
        plt.ylabel('theta dot')
        plt.title('Fit Value Function')
        plt.show()

